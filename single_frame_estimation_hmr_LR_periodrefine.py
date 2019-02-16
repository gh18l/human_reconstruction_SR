import tensorflow as tf
from camera import Perspective_Camera
import numpy as np
from smpl_batch import SMPL
import util
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from opendr_render import render
import os
import cv2

def refine_optimization(poses, betas, trans, data_dict, LR_cameras, texture_img, texture_vt):
    LR_j2ds = data_dict["j2ds"]
    LR_confs = data_dict["confs"]
    LR_j2ds_face = data_dict["j2ds_face"]
    LR_confs_face = data_dict["confs_face"]
    LR_j2ds_head = data_dict["j2ds_head"]
    LR_confs_head = data_dict["confs_head"]
    LR_j2ds_foot = data_dict["j2ds_foot"]
    LR_confs_foot = data_dict["confs_foot"]
    LR_imgs = data_dict["imgs"]
    LR_masks = data_dict["masks"]
    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)
    j3ds_old = []
    videowriter = []
    if util.video == True:
        fps = 15
        size = (LR_imgs[0].shape[1], LR_imgs[0].shape[0])
        video_path = util.hmr_path + "output_after_refine/texture.mp4"
        videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    pose_final_old = []
    for ind in range(len(poses)):
        print("The refine %d iteration" % ind)
        param_shape = tf.Variable(betas[ind, :].reshape([1, -1]), dtype=tf.float32)
        param_rot = tf.Variable(poses[ind, 0:3].reshape([1, -1]), dtype=tf.float32)
        param_pose = tf.Variable(poses[ind, 3:72].reshape([1, -1]), dtype=tf.float32)
        param_trans = tf.Variable(trans[ind, :].reshape([1, -1]), dtype=tf.float32)
        initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
        cam_LR = Perspective_Camera(LR_cameras[ind][0], LR_cameras[ind][0], LR_cameras[ind][1],
                                    LR_cameras[ind][2], LR_cameras[ind][3], np.zeros(3))
        j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
        j3ds = tf.reshape(j3ds, [-1, 3])
        v = tf.reshape(v, [-1, 3])
        jointsplus = tf.reshape(jointsplus, [-1, 3])
        j2ds_est = cam_LR.project(tf.squeeze(j3ds))
        j2dsplus_est = cam_LR.project(tf.squeeze(jointsplus))

        objs = {}
        base_weights = 1.0 * np.array(
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        weights = LR_confs[ind] * base_weights
        weights = tf.constant(weights, dtype=tf.float32)
        objs['J2D_Loss'] = tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_est[2:, :] - LR_j2ds[ind]), 1))

        base_weights_face = 1.5 * np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0])
        weights_face = LR_confs_face[ind] * base_weights_face
        weights_face = tf.constant(weights_face, dtype=tf.float32)
        objs['J2D_face_Loss'] = tf.reduce_sum(
            weights_face * tf.reduce_sum(tf.square(j2dsplus_est[14:19, :] - LR_j2ds_face[ind]), 1))

        base_weights_head = 1.0 * np.array(
            [1.0, 1.0])
        weights_head = LR_confs_head[ind] * base_weights_head
        weights_head = tf.constant(weights_head, dtype=tf.float32)
        objs['J2D_head_Loss'] = tf.reduce_sum(
            weights_head * tf.reduce_sum(tf.square(LR_j2ds_head[ind] - j2ds_est[14:16, :]), 1))

        base_weights_foot = 1.0 * np.array(
            [1.0, 1.0])
        _LR_confs_foot = np.zeros(2)
        if LR_confs_foot[ind][0] != 0 and LR_confs_foot[ind][1] != 0:
            _LR_confs_foot[0] = (LR_confs_foot[ind][0] + LR_confs_foot[ind][1]) / 2.0
        else:
            _LR_confs_foot[0] = 0.0
        if LR_confs_foot[ind][3] != 0 and LR_confs_foot[ind][4] != 0:
            _LR_confs_foot[1] = (LR_confs_foot[ind][3] + LR_confs_foot[ind][4]) / 2.0
        else:
            _LR_confs_foot[1] = 0.0
        weights_foot = _LR_confs_foot * base_weights_foot
        weights_foot = tf.constant(weights_foot, dtype=tf.float32)
        _LR_j2ds_foot = np.zeros([2, 2])
        _LR_j2ds_foot[0, 0] = (LR_j2ds_foot[ind][0, 0] + LR_j2ds_foot[ind][1, 0]) / 2.0
        _LR_j2ds_foot[0, 1] = (LR_j2ds_foot[ind][0, 1] + LR_j2ds_foot[ind][1, 1]) / 2.0
        _LR_j2ds_foot[1, 0] = (LR_j2ds_foot[ind][3, 0] + LR_j2ds_foot[ind][4, 0]) / 2.0
        _LR_j2ds_foot[1, 1] = (LR_j2ds_foot[ind][3, 1] + LR_j2ds_foot[ind][4, 1]) / 2.0
        objs['J2D_foot_Loss'] = tf.reduce_sum(
            weights_foot * tf.reduce_sum(tf.square(_LR_j2ds_foot - j2ds_est[0:2, :]), 1))

        param_complete_pose = tf.concat([param_rot, param_pose], axis=1)
        objs['Prior_Loss'] = 200.0 * tf.reduce_sum(tf.square(param_complete_pose[0, :] - poses[ind, :]))
        objs['Prior_Shape'] = 5.0 * tf.reduce_sum(tf.square(param_shape))
        w1 = np.array([1.04 * 2.0, 1.04 * 2.0, 57.4 * 50, 57.4 * 50])
        w1 = tf.constant(w1, dtype=tf.float32)
        objs["angle_elbow_knee"] = 0.005 * tf.reduce_sum(w1 * [
            tf.exp(param_pose[0, 52]), tf.exp(-param_pose[0, 55]),
            tf.exp(-param_pose[0, 9]), tf.exp(-param_pose[0, 12])])
        w_temporal = [0.5, 0.5, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 7.0]
        if ind != 0:
            objs['temporal'] = 100.0 * tf.reduce_sum(
                w_temporal * tf.reduce_sum(tf.square(j3ds - j3ds_old), 1))
            objs['temporal_pose'] = 0.0 * tf.reduce_sum(
                tf.square(pose_final_old[0, 3:72] - param_pose[0, :]))
            #objs['temporal_pose_rot'] = 10000.0 * tf.reduce_sum(
                #tf.square(pose_final_old[0, 0:3] - param_rot[0, :]))
        loss = tf.reduce_mean(objs.values())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #L-BFGS-B
            optimizer = scipy_pt(loss=loss,
                        var_list=[param_rot, param_trans, param_shape],
                    options={'eps': 1e-12, 'ftol': 1e-12, 'maxiter': 500, 'disp': False})
            optimizer.minimize(sess)
            pose_final, betas_final, trans_final = sess.run(
                [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])
            v_final = sess.run([v, j3ds])
            cam_LR1 = sess.run([cam_LR.fl_x, cam_LR.cx, cam_LR.cy, cam_LR.trans])
            camera = render.camera(cam_LR1[0], cam_LR1[1], cam_LR1[2], cam_LR1[3])
            img_result_texture = camera.render_texture(v_final[0], texture_img, texture_vt)
            if not os.path.exists(util.hmr_path + "output_after_refine"):
                os.makedirs(util.hmr_path + "output_after_refine")
            cv2.imwrite(util.hmr_path + "output_after_refine/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
            if util.video is True:
                videowriter.write(img_result_texture)
            img_result_naked = camera.render_naked(v_final[0], LR_imgs[ind])
            cv2.imwrite(util.hmr_path + "output_after_refine/hmr_optimization_%04d.png" % ind, img_result_naked)
            img_result_naked_rotation = camera.render_naked_rotation(v_final[0], 90, LR_imgs[ind])
            cv2.imwrite(util.hmr_path + "output_after_refine/hmr_optimization_rotation_%04d.png" % ind, img_result_naked_rotation)
            _objs = sess.run(objs)
            print("the refine j2d loss is %f" % _objs['J2D_Loss'])
            print("the refine J2D_face loss is %f" % _objs['J2D_face_Loss'])
            print("the refine J2D_head loss is %f" % _objs['J2D_head_Loss'])
            print("the refine J2D_foot loss is %f" % _objs['J2D_foot_Loss'])
            print("the refine prior loss is %f" % _objs['Prior_Loss'])
            print("the refine Prior_Shape loss is %f" % _objs['Prior_Shape'])
            print("the refine angle_elbow_knee loss is %f" % _objs["angle_elbow_knee"])

            j3ds_old = v_final[1]
            pose_final_old = pose_final