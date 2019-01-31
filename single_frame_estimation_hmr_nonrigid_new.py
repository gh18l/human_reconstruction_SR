import util
import tensorflow as tf
from smpl_batch import SMPL
import numpy as np
from camera import Perspective_Camera
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
import os
import cv2
from opendr_render import render
import optimization_prepare as opt_pre
ind = 22

##point -- 1*2 points -- 6890*2
def get_distance(point, points):
    distances = np.zeros([6890])
    for i in range(len(points)):
        distance = np.square(point[0, 0] - points[i, 0]) + np.square(point[0, 1] - points[i, 1])
        distances[i] = distance
    return distances

def nonrigid_estimation():
    hmr_dict, data_dict = util.load_hmr_data(util.hmr_path)
    hmr_thetas = hmr_dict["hmr_thetas"]
    hmr_betas = hmr_dict["hmr_betas"]
    hmr_trans = hmr_dict["hmr_trans"]
    hmr_cams = hmr_dict["hmr_cams"]
    hmr_joint3ds = hmr_dict["hmr_joint3ds"]

    HR_j2ds = data_dict["j2ds"]
    HR_confs = data_dict["confs"]
    HR_j2ds_face = data_dict["j2ds_face"]
    HR_confs_face = data_dict["confs_face"]
    HR_j2ds_head = data_dict["j2ds_head"]
    HR_confs_head = data_dict["confs_head"]
    HR_j2ds_foot = data_dict["j2ds_foot"]
    HR_confs_foot = data_dict["confs_foot"]
    HR_imgs = data_dict["imgs"]
    HR_masks = data_dict["masks"]

    initial_param, pose_mean, pose_covariance = util.load_initial_param()
    pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)
    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)
    j3ds_old = []
    pose_final_old = []
    pose_final = []

    ####optimization
    hmr_theta = hmr_thetas[ind, :].squeeze()
    hmr_shape = hmr_betas[ind, :].squeeze()
    hmr_tran = hmr_trans[ind, :].squeeze()
    ### fix camera
    hmr_cam = hmr_cams[ind, :].squeeze()

    hmr_joint3d = hmr_joint3ds[ind, :, :]
    # print("the %d arm error is %f" % (ind, np.fabs((hmr_joint3d[6, 2] + hmr_joint3d[7, 2]) - (hmr_joint3d[10, 2] + hmr_joint3d[11, 2]))))
    arm_error = np.fabs((hmr_joint3d[6, 2] + hmr_joint3d[7, 2]) - (hmr_joint3d[10, 2] + hmr_joint3d[11, 2]))
    leg_error = np.fabs((hmr_joint3d[0, 2] + hmr_joint3d[1, 2]) - (hmr_joint3d[5, 2] + hmr_joint3d[4, 2]))
    ####arm
    ####leg
    if leg_error > 0.1:
        if hmr_joint3d[0, 2] + hmr_joint3d[1, 2] < hmr_joint3d[5, 2] + hmr_joint3d[4, 2]:
            hmr_theta[51] = 0.8
            hmr_theta[52] = 1e-8
            hmr_theta[53] = 1.0
            hmr_theta[58] = 1e-8
            forward_arm = "left"
        else:
            hmr_theta[48] = 0.8
            hmr_theta[49] = 1e-8
            hmr_theta[50] = -1.0
            hmr_theta[55] = 1e-8
            forward_arm = "right"
    #####arm
    else:
        if hmr_joint3d[6, 2] + hmr_joint3d[7, 2] < hmr_joint3d[10, 2] + hmr_joint3d[11, 2]:
            hmr_theta[48] = 0.8
            hmr_theta[49] = 1e-8
            hmr_theta[50] = -1.0
            hmr_theta[55] = 1e-8
            forward_arm = "right"
        else:
            hmr_theta[51] = 0.8
            hmr_theta[52] = 1e-8
            hmr_theta[53] = 1.0
            hmr_theta[58] = 1e-8
            forward_arm = "left"
    print(forward_arm)

    ####numpy array initial_param
    initial_param_np = np.concatenate(
        [hmr_shape.reshape([1, -1]), hmr_theta.reshape([1, -1]), hmr_tran.reshape([1, -1])], axis=1)

    param_shape = tf.Variable(hmr_shape.reshape([1, -1]), dtype=tf.float32)
    param_rot = tf.Variable(hmr_theta[0:3].reshape([1, -1]), dtype=tf.float32)
    param_pose = tf.Variable(hmr_theta[3:72].reshape([1, -1]), dtype=tf.float32)
    param_trans = tf.Variable(hmr_tran.reshape([1, -1]), dtype=tf.float32)

    ####tensorflow array initial_param_tf
    initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
    # cam_HR, camera_t_final_HR = initialize_camera(smpl_model, HR_j2ds[0], HR_imgs[0], initial_param_np, flength)
    cam_HR = Perspective_Camera(hmr_cam[0], hmr_cam[0], hmr_cam[1],
                                hmr_cam[2], np.zeros(3), np.zeros(3))
    j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
    j3ds = tf.reshape(j3ds, [-1, 3])
    jointsplus = tf.reshape(jointsplus, [-1, 3])
    hmr_joint3d = tf.constant(hmr_joint3d.reshape([-1, 3]), dtype=tf.float32)
    v = tf.reshape(v, [-1, 3])
    j2ds_est = []
    verts_est = []
    j2ds_est = cam_HR.project(tf.squeeze(j3ds))
    j2dsplus_est = cam_HR.project(tf.squeeze(jointsplus))
    verts_est = cam_HR.project(tf.squeeze(v))
    j2ds_est = tf.convert_to_tensor(j2ds_est)
    ####mask
    HR_mask = tf.convert_to_tensor(HR_masks[ind], dtype=tf.float32)
    verts2dsilhouette = opt_pre.get_tf_mask(verts_est, HR_masks[ind])

    objs = {}
    base_weights = np.array(
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  #######
    weights = HR_confs[ind] * base_weights
    weights = tf.constant(weights, dtype=tf.float32)
    objs['J2D_Loss'] = 1.0 * tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_est[2:, :] - HR_j2ds[ind]), 1))

    base_weights_face = 2.5 * np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0])
    weights_face = HR_confs_face[ind] * base_weights_face
    weights_face = tf.constant(weights_face, dtype=tf.float32)
    objs['J2D_face_Loss'] = tf.reduce_sum(
        weights_face * tf.reduce_sum(tf.square(j2dsplus_est[14:19, :] - HR_j2ds_face[ind]), 1))
    # objs['J2D_face_Loss'] = 10000000.0 * tf.reduce_sum(
    # tf.square(j2dsplus_est[14, :] - HR_j2ds_face[ind][0, :]))

    base_weights_head = 1.0 * np.array(
        [1.0, 1.0])
    weights_head = HR_confs_head[ind] * base_weights_head
    weights_head = tf.constant(weights_head, dtype=tf.float32)
    objs['J2D_head_Loss'] = tf.reduce_sum(
        weights_head * tf.reduce_sum(tf.square(HR_j2ds_head[ind] - j2ds_est[14:16, :]), 1))

    base_weights_foot = 1.0 * np.array(
        [1.0, 1.0])
    _HR_confs_foot = np.zeros(2)
    if HR_confs_foot[ind][0] != 0 and HR_confs_foot[ind][1] != 0:
        _HR_confs_foot[0] = (HR_confs_foot[ind][0] + HR_confs_foot[ind][1]) / 2.0
    else:
        _HR_confs_foot[0] = 0.0
    if HR_confs_foot[ind][3] != 0 and HR_confs_foot[ind][4] != 0:
        _HR_confs_foot[1] = (HR_confs_foot[ind][3] + HR_confs_foot[ind][4]) / 2.0
    else:
        _HR_confs_foot[1] = 0.0
    weights_foot = _HR_confs_foot * base_weights_foot
    weights_foot = tf.constant(weights_foot, dtype=tf.float32)
    _HR_j2ds_foot = np.zeros([2, 2])
    _HR_j2ds_foot[0, 0] = (HR_j2ds_foot[ind][0, 0] + HR_j2ds_foot[ind][1, 0]) / 2.0
    _HR_j2ds_foot[0, 1] = (HR_j2ds_foot[ind][0, 1] + HR_j2ds_foot[ind][1, 1]) / 2.0
    _HR_j2ds_foot[1, 0] = (HR_j2ds_foot[ind][3, 0] + HR_j2ds_foot[ind][4, 0]) / 2.0
    _HR_j2ds_foot[1, 1] = (HR_j2ds_foot[ind][3, 1] + HR_j2ds_foot[ind][4, 1]) / 2.0
    objs['J2D_foot_Loss'] = tf.reduce_sum(
        weights_foot * tf.reduce_sum(tf.square(_HR_j2ds_foot - j2ds_est[0:2, :]), 1))

    pose_diff = tf.reshape(param_pose - pose_mean, [1, -1])
    objs['Prior_Loss'] = 1.0 * tf.squeeze(tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))
    objs['Prior_Shape'] = 5.0 * tf.reduce_sum(tf.square(param_shape))
    ##############################################################
    ##########control the angle of the elbow and knee#############
    ##############################################################
    # w1 = np.array([4.04 * 1e2, 4.04 * 1e2, 57.4 * 10, 57.4 * 10])
    w1 = np.array([1.04 * 2.0, 1.04 * 2.0, 57.4 * 50, 57.4 * 50])
    w1 = tf.constant(w1, dtype=tf.float32)
    objs["angle_elbow_knee"] = 0.03 * tf.reduce_sum(w1 * [
        tf.exp(param_pose[0, 52]), tf.exp(-param_pose[0, 55]),
        tf.exp(-param_pose[0, 9]), tf.exp(-param_pose[0, 12])])

    ##############################################################
    ###################mask obj###################################
    ##############################################################
    objs['mask'] = 0.05 * tf.reduce_sum(verts2dsilhouette / 255.0 * (255.0 - HR_mask) / 255.0
                                        + (255.0 - verts2dsilhouette) / 255.0 * HR_mask / 255.0)

    loss = tf.reduce_mean(objs.values())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # L-BFGS-B
        optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_trans, param_pose, param_shape],
                             options={'eps': 1e-12, 'ftol': 1e-12, 'maxiter': 1000, 'disp': False}, method='L-BFGS-B')
        optimizer.minimize(sess)
        cam_HR1 = sess.run([cam_HR.fl_x, cam_HR.cx, cam_HR.cy, cam_HR.trans])
        #if not os.path.exists(util.hmr_path + "output/camera_after_optimization.npy"):
            #np.save(util.hmr_path + "output/camera_after_optimization.npy", cam_HR1)
        v_final = sess.run([v, verts_est, j3ds])
        camera = render.camera(cam_HR1[0], cam_HR1[1], cam_HR1[2], cam_HR1[3])
        uv_real, vt = camera.generate_uv(v_final[0], HR_imgs[ind])
        if not os.path.exists(util.hmr_path + "output_nonrigid"):
            os.makedirs(util.hmr_path + "output_nonrigid")
        if util.crop_texture is True:
            img_result_texture, HR_mask_img = camera.render_texture(v_final[0], HR_imgs[ind], vt, HR_masks[ind])
            if ind == 4:
                if not os.path.exists(util.texture_path):
                    os.makedirs(util.texture_path)
                camera.write_texture_data(util.texture_path, HR_mask_img, vt)
        else:
            img_result_texture, _ = camera.render_texture(v_final[0], HR_imgs[ind], vt)
            if ind == 4:
                if not os.path.exists(util.texture_path):
                    os.makedirs(util.texture_path)
                camera.write_texture_data(util.texture_path, HR_imgs[ind], vt)
        cv2.imwrite(util.hmr_path + "output_nonrigid/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
        img_result_naked = camera.render_naked(v_final[0], HR_imgs[ind])
        cv2.imwrite(util.hmr_path + "output_nonrigid/hmr_optimization_%04d.png" % ind, img_result_naked)
        img_result_naked_rotation = camera.render_naked_rotation(v_final[0], 90, HR_imgs[ind])
        cv2.imwrite(util.hmr_path + "output_nonrigid/hmr_optimization_rotation_%04d.png" % ind, img_result_naked_rotation)

        pose_final, betas_final, trans_final = sess.run(
            [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])

        # model_f = sess.run(smpl_model.f)
        _objs = sess.run(objs)
        for name in _objs:
            print("the HR %s loss is %f" % (name,_objs[name]))
        # print("the HR J2D_face loss is %f" % _objs['J2D_face_Loss'])
        # print("the HR J2D_head loss is %f" % _objs['J2D_head_Loss'])
        # print("the HR J2D_foot loss is %f" % _objs['J2D_foot_Loss'])
        # print("the HR prior loss is %f" % _objs['Prior_Loss'])
        # print("the HR Prior_Shape loss is %f" % _objs['Prior_Shape'])
        # print("the HR angle_elbow_knee loss is %f" % _objs["angle_elbow_knee"])
        # print("the HR mask loss is %f" % _objs['mask'])
        # print("the arm_leg_direction loss is %f" % sess.run(objs["arm_leg_direction"]))
        # model_f = model_f.astype(int).tolist()
        #pose_final, betas_final, trans_final = sess.run(
            #[tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])
    verts2d = v_final[1]
    for z in range(len(verts2d)):
        if int(verts2d[z][0]) > HR_masks[ind].shape[0] - 1:
            print(int(verts2d[z][0]))
            verts2d[z][0] = HR_masks[ind].shape[0] - 1
        if int(verts2d[z][1]) > HR_masks[ind].shape[1] - 1:
            print(int(verts2d[z][1]))
            verts2d[z][1] = HR_masks[ind].shape[1] - 1
        (HR_masks[ind])[int(verts2d[z][0]), int(verts2d[z][1])] = 127
    if not os.path.exists(util.hmr_path + "output_mask"):
        os.makedirs(util.hmr_path + "output_mask")
    cv2.imwrite(util.hmr_path + "output_mask/%04d.png" % ind, HR_masks[ind])




    contours, hierarchy = cv2.findContours(HR_masks[ind], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_num = np.zeros(len(contours))
    for i in range(len(contours)):
        contours_num[i] = len(contours[i])
    contours_index = contours_num.argmax()

    contours_smpl_index = np.zeros(len(contours[contours_index]))

    for i in range(len(contours[contours_index])):
        distances = get_distance(contours[contours_index][i, :, :], uv_real)
        min_smpl_index = np.argmin(distances)
        contours_smpl_index[i] = min_smpl_index

    ##### generate smpl shape template
    param_shape = tf.Variable(betas_final.reshape([1, -1]), dtype=tf.float32)
    param_rot = tf.constant(pose_final[0:3].reshape([1, -1]), dtype=tf.float32)
    param_pose = tf.constant(pose_final[3:72].reshape([1, -1]), dtype=tf.float32)
    param_trans = tf.constant(trans_final.reshape([1, -1]), dtype=tf.float32)

    ####tensorflow array initial_param_tf
    initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
    j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
    v_shaped_tf = tf.reshape(v, [-1, 3])

    v = v_final[0]
    v_tf = tf.Variable(v, dtype=tf.float32)

    ### convert v_tf to laplace coordination
    L = opt_pre.get_laplace_operator(v_shaped_tf, smpl_model.f)
    delta = tf.matmul(L, v_shaped_tf)
    weights_laplace = opt_pre.get_laplace_weights()
    weights_laplace = 4.0 * weights_laplace.reshape(-1, 1)


    cam_nonrigid = Perspective_Camera(cam_HR1[0], cam_HR1[0], hmr_cam[1],
                                      hmr_cam[2], cam_HR1[3], np.zeros(3))
    verts_est = cam_nonrigid.project(tf.squeeze(v_tf))
    objs_nonrigid = {}
    objs_nonrigid['verts_loss'] = 1.0 * tf.reduce_sum(tf.square(verts_est[contours_smpl_index, :] - contours[contours_index].squeeze()))
    #### norm choose
    objs_nonrigid['laplace'] = 1.0 * tf.reduce_sum(weights_laplace * tf.reduce_sum(tf.matmul(L, v_tf) - delta, 1))
    objs_nonrigid['smooth_loss'] = 1.0 * tf.reduce_sum(tf.square(smpl_contours_tf - smpl_contours))