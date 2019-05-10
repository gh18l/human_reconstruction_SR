import tensorflow as tf
from camera import Perspective_Camera
import numpy as np
from smpl_batch import SMPL
import util
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from opendr_render import render
import os
import cv2
import optimization_prepare as opt_pre
import smpl_np
import correct_final_texture as tex
import pickle

def load_pose_pkl():
    LR_path = util.hmr_path + "output"
    LR_pkl_files = os.listdir(LR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    j3dss = []
    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        j3ds = param['j3ds']
        j3dss.append(j3ds)
    return j3dss

def refine_optimization(poses, betas, trans, data_dict, hmr_dict, LR_cameras, texture_img, texture_vt):
    j3dss = load_pose_pkl()
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

    hmr_thetas = hmr_dict["hmr_thetas"]
    hmr_betas = hmr_dict["hmr_betas"]
    hmr_trans = hmr_dict["hmr_trans"]
    hmr_cams = hmr_dict["hmr_cams"]
    hmr_joint3ds = hmr_dict["hmr_joint3ds"]

    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)
    body_parsing_idx = opt_pre.load_body_parsing()
    j3ds_old = []
    verts_body_old = []
    videowriter = []
    if util.video == True:
        fps = 15
        size = (LR_imgs[0].shape[1], LR_imgs[0].shape[0])
        video_path = util.hmr_path + "output_after_refine/texture.mp4"
        videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    pose_final_old = []
    for ind in range(len(poses)):
        hmr_theta = hmr_thetas[ind, :].squeeze()
        hmr_shape = hmr_betas[ind, :].squeeze()
        hmr_tran = hmr_trans[ind, :].squeeze()
        hmr_cam = hmr_cams[ind, :].squeeze()
        if util.pedestrian_constraint == True:
            posepre_joint3d = j3dss[ind]
            if posepre_joint3d[2, 2] < posepre_joint3d[7, 2]:
                poses[ind, 51] = 0.8
                poses[ind, 52] = 1e-8
                poses[ind, 53] = 1.0
                poses[ind, 58] = 1e-8
                forward_arm = "left"
            else:
                poses[ind, 48] = 0.8
                poses[ind, 49] = 1e-8
                poses[ind, 50] = -1.0
                poses[ind, 55] = 1e-8
                forward_arm = "right"
            print(forward_arm)
        print("The refine %d iteration" % ind)
        param_shape = tf.Variable(hmr_shape.reshape([1, -1]), dtype=tf.float32)
        param_rot = tf.Variable(hmr_theta[0:3].reshape([1, -1]), dtype=tf.float32)
        param_pose = tf.Variable(hmr_theta[3:72].reshape([1, -1]), dtype=tf.float32)
        param_trans = tf.Variable(hmr_tran.reshape([1, -1]), dtype=tf.float32)

        ###to get hmr 2d verts
        param_shape_fixed = tf.constant(hmr_shape.reshape([1, -1]), dtype=tf.float32)
        hmr_param_rot_fixed = tf.constant(hmr_theta[0:3].reshape([1, -1]), dtype=tf.float32)
        hmr_param_pose_fixed = tf.constant(hmr_theta[3:72].reshape([1, -1]), dtype=tf.float32)
        hmr_param_trans_fixed = tf.constant(hmr_tran.reshape([1, -1]), dtype=tf.float32)

        initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
        initial_param_tf_fixed = tf.concat(
            [param_shape_fixed, hmr_param_rot_fixed, hmr_param_pose_fixed, hmr_param_trans_fixed], axis=1)
        cam_LR = Perspective_Camera(LR_cameras[ind][0], LR_cameras[ind][0], LR_cameras[ind][1],
                                    LR_cameras[ind][2], LR_cameras[ind][3], np.zeros(3))
        j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
        _, v_hmr_fixed, __ = smpl_model.get_3d_joints(initial_param_tf_fixed, util.SMPL_JOINT_IDS)
        j3ds = tf.reshape(j3ds, [-1, 3])
        v = tf.reshape(v, [-1, 3])
        v_hmr_fixed = tf.reshape(v_hmr_fixed, [-1, 3])
        jointsplus = tf.reshape(jointsplus, [-1, 3])
        j2ds_est = cam_LR.project(tf.squeeze(j3ds))
        j2dsplus_est = cam_LR.project(tf.squeeze(jointsplus))
        verts_est = cam_LR.project(tf.squeeze(v))
        verts_est_fixed = cam_LR.project(tf.squeeze(v_hmr_fixed))

        objs = {}
        base_weights = 1.0 * np.array(
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        weights = LR_confs[ind] * base_weights
        weights = tf.constant(weights, dtype=tf.float32)
        objs['J2D_Loss'] = tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_est[2:, :] - LR_j2ds[ind]), 1))

        base_weights_face = 2.5 * np.array(
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
        ### 6000.0
        #weights = np.ones([1, 72])
        #weights[0,54] = 0
        #weights[0,55] = 0
        #weights[0,56] = 0
        #weights[0, 48] = 0
        #weights[0, 49] = 0
        #weights[0, 50] = 0
        #weights = tf.constant(weights, dtype=tf.float32)
        objs['Prior_Loss'] = 6000.0 * tf.reduce_sum(tf.square(param_complete_pose[0, :] - poses[ind, :]))
        objs['Prior_Shape'] = 5.0 * tf.reduce_sum(tf.square(param_shape))
        w1 = np.array([1.04 * 2.0, 1.04 * 2.0, 57.4 * 50, 57.4 * 50])
        w1 = tf.constant(w1, dtype=tf.float32)
        objs["angle_elbow_knee"] = 0.005 * tf.reduce_sum(w1 * [
            tf.exp(param_pose[0, 52]), tf.exp(-param_pose[0, 55]),
            tf.exp(-param_pose[0, 9]), tf.exp(-param_pose[0, 12])])
        w_temporal = [0.5, 0.5, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 7.0]

        param_pose_full = tf.concat([param_rot, param_pose], axis=1)
        objs['hmr_constraint'] = 0.0 * tf.reduce_sum(tf.square(tf.squeeze(param_pose_full) - hmr_theta))
        ### 8000.0
        objs['hmr_hands_constraint'] = 0.0 * tf.reduce_sum(
            tf.square(tf.squeeze(param_pose_full)[21] - hmr_theta[21])
            + tf.square(tf.squeeze(param_pose_full)[23] - hmr_theta[23])
            + tf.square(tf.squeeze(param_pose_full)[20] - hmr_theta[20])
            + tf.square(tf.squeeze(param_pose_full)[22] - hmr_theta[22]))

        #param_pose_full = tf.concat([param_rot, param_pose], axis=1)
        #objs['hmr_constraint'] = 30.0 * tf.reduce_sum(tf.square(tf.squeeze(param_pose_full) - hmr_theta))
        if ind != 0:
            w_temporal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10000.0, 0.0, 0.0, 0.0]
            objs['temporal'] = 0.0 * tf.reduce_sum(
                w_temporal * tf.reduce_sum(tf.square(j3ds - j3ds_old), 1))
            p_weights = np.ones([1, 69])
            # p_weights[0, 51] = 10000
            # p_weights[0, 52] = 10000
            # p_weights[0, 53] = 10000
            p_weights = tf.constant(p_weights, dtype=tf.float32)
            objs['temporal_pose'] = 0.0 * tf.reduce_sum(p_weights *
                tf.square(pose_final_old[0, 3:72] - param_pose[0, :]))
            ##optical flow constraint
            body_idx = np.array(np.hstack([body_parsing_idx[0],
                                           body_parsing_idx[2]])).squeeze()
            body_idx = body_idx.reshape([-1, 1]).astype(np.int64)
            verts_est_body = tf.gather_nd(verts_est, body_idx)
            objs['dense_optflow'] = 0.0 * tf.reduce_sum(tf.square(
                verts_est_body - verts_body_old))
        loss = tf.reduce_mean(objs.values())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #L-BFGS-B
            optimizer = scipy_pt(loss=loss,
                        var_list=[param_trans, param_pose, param_rot],
                    options={'eps': 1e-12, 'ftol': 1e-12, 'maxiter': 500, 'disp': False})
            optimizer.minimize(sess)
            pose_final, betas_final, trans_final = sess.run(
                [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])
            #betas_final = betas_final * 0.05
            v_final = sess.run([v, verts_est, j3ds])
            v_final_fixed = sess.run(verts_est_fixed)
            cam_LR1 = sess.run([cam_LR.fl_x, cam_LR.cx, cam_LR.cy, cam_LR.trans])
            camera = render.camera(cam_LR1[0], cam_LR1[1], cam_LR1[2], cam_LR1[3])

            ### set nonrigid template
            smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
            template = np.load(util.texture_path + "template.npy")
            smpl.set_template(template)
            v = smpl.get_verts(pose_final, betas_final, trans_final)

            img_result_texture = camera.render_texture(v, texture_img, texture_vt)
            #img_result_texture = tex.correct_render_small(img_result_texture, 3)
            if not os.path.exists(util.hmr_path + "output_after_refine"):
                os.makedirs(util.hmr_path + "output_after_refine")
            cv2.imwrite(util.hmr_path + "output_after_refine/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
            img_bg = cv2.resize(LR_imgs[ind], (util.img_width, util.img_height))
            img_result_texture_bg = camera.render_texture_imgbg(img_result_texture, img_bg)
            cv2.imwrite(util.hmr_path + "output_after_refine/texture_bg_%04d.png" % ind,
                        img_result_texture_bg)
            if util.video is True:
                videowriter.write(img_result_texture)
            img_result_naked = camera.render_naked(v, LR_imgs[ind])
            cv2.imwrite(util.hmr_path + "output_after_refine/hmr_optimization_%04d.png" % ind, img_result_naked)
            img_result_naked_rotation = camera.render_naked_rotation(v, 90, LR_imgs[ind])
            cv2.imwrite(util.hmr_path + "output_after_refine/hmr_optimization_rotation_%04d.png" % ind, img_result_naked_rotation)
            _objs = sess.run(objs)
            for name in _objs:
                print("the %s loss is %f" % (name, _objs[name]))
            j3ds_old = v_final[2]
            pose_final_old = pose_final

            res = {'pose': pose_final, 'betas': betas_final, 'trans': trans_final, 'j3ds': v_final[2], 'cam': cam_LR1}
            # out_pkl_path = out_ply_path.replace('.ply', '.pkl')
            with open(util.hmr_path + "output_after_refine/hmr_optimization_pose_%04d.pkl" % ind, 'wb') as fout:
                pickle.dump(res, fout)

            #### get body vertex corresponding 2d index
            if ind != len(LR_j2ds) - 1:
                if ind == 0:  ###first is confident in hmr
                    body_idx = np.array(body_parsing_idx[0]).squeeze()  ##body part
                    head_idx = np.array(body_parsing_idx[1]).squeeze()
                    leg_idx = np.array(body_parsing_idx[2]).squeeze()
                    verts_body = np.vstack([v_final_fixed[body_idx],
                                        v_final_fixed[leg_idx]])
                    verts_body_old = opt_pre.get_dense_correspondence(verts_body,
                                                                      LR_imgs[ind], LR_imgs[ind + 1])
                else:
                    body_idx = np.array(body_parsing_idx[0]).squeeze()  ##body part
                    head_idx = np.array(body_parsing_idx[1]).squeeze()
                    leg_idx = np.array(body_parsing_idx[2]).squeeze()
                    verts_body = np.vstack([v_final[1][body_idx],
                                        v_final[1][leg_idx]])
                    verts_body_old = opt_pre.get_dense_correspondence(verts_body,
                                                                      LR_imgs[ind], LR_imgs[ind + 1])