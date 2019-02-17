import pickle
import util
import numpy as np
import tensorflow as tf
from camera import Perspective_Camera
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
import os
import cv2
from smpl_batch_body_parsing import SMPL
from opendr_render import render
import pickle as pkl
dd = pickle.load(open(util.NORMAL_SMPL_PATH))
weights = dd['weights']
vert_sym_idxs = dd['vert_sym_idxs']
v_template = dd['v_template']
leg_index = [1, 4, 7, 10, 2, 5, 8, 11]
arm_index = [17, 19, 21, 23, 16, 18, 20, 22, 14, 13]
body_index = [6]
head_index = [12, 15]
body_parsing_idx = []  ###body head
_leg_idx = np.zeros(6890)
_arm_idx = np.zeros(6890)
_body_idx = np.zeros(6890)
_head_idx = np.zeros(6890)
placeholder_idx = np.zeros(6890)
_test_idx = np.zeros(6890)

for _, iii in enumerate(body_index):
    length = len(weights[:, iii])
    for ii in range(length):
        if weights[ii, iii] > 0.6 and placeholder_idx[ii] == 0:
            _body_idx[ii] = 1
            placeholder_idx[ii] = 1
            _test_idx[ii] = 1
body_idx = np.where(_body_idx == 1)
body_parsing_idx.append(body_idx)

for _, iii in enumerate(head_index):
    length = len(weights[:, iii])
    for ii in range(length):
        if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
            _head_idx[ii] = 1
            placeholder_idx[ii] = 1
            _test_idx[ii] = 1
head_idx = np.where(_head_idx == 1)
body_parsing_idx.append(head_idx)

for _, iii in enumerate(leg_index):
    length = len(weights[:, iii])
    for ii in range(length):
        if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
            _leg_idx[ii] = 1
            placeholder_idx[ii] = 1
            _test_idx[ii] = 1
leg_idx = np.where(_leg_idx == 1)
body_parsing_idx.append(leg_idx)

for _, iii in enumerate(arm_index):
    length = len(weights[:, iii])
    for ii in range(length):
        if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
            _arm_idx[ii] = 1
            placeholder_idx[ii] = 1
            _test_idx[ii] = 1
arm_idx = np.where(_arm_idx == 1)
body_parsing_idx.append(arm_idx)

with open("./smpl/models/bodyparts.pkl",'rb') as f:
    v_ids = pkl.load(f)
#test
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
#ax = plt.subplot(111)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(v_template[body_parsing_idx[0], 0], v_template[body_parsing_idx[0], 1], v_template[body_parsing_idx[0], 2], c='b')
#ax.scatter(v_template[v_ids['hand_l'], 0], v_template[v_ids['hand_l'], 1], v_template[v_ids['hand_l'], 2], c='g')
#ax.scatter(v_template[v_ids['hand_r'], 0], v_template[v_ids['hand_r'], 1], v_template[v_ids['hand_r'], 2], c='y')
#ax.scatter(v_template[v_ids['fingers_l'], 0], v_template[v_ids['fingers_l'], 1], v_template[v_ids['fingers_l'], 2], c='c')
ax.scatter(v_template[:, 0], v_template[:, 1], v_template[:, 2], c='r', s=1)
plt.show()

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
smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH, body_parsing_idx)
for ind, HR_j2d in enumerate(HR_j2ds):
    print("the shape %d iteration" % ind)
    hmr_theta = hmr_thetas[ind, :].squeeze()
    hmr_shape = hmr_betas[ind, :].squeeze()
    hmr_tran = hmr_trans[ind, :].squeeze()
    hmr_cam = hmr_cams[ind, :].squeeze()

    ### align beta parameter
    hmr_shape_parsing = np.row_stack((hmr_shape, hmr_shape, hmr_shape, hmr_shape))

    param_shape = tf.Variable(hmr_shape_parsing.reshape([1, -1]), dtype=tf.float32)
    param_rot = tf.Variable(hmr_theta[0:3].reshape([1, -1]), dtype=tf.float32)
    param_pose = tf.Variable(hmr_theta[3:72].reshape([1, -1]), dtype=tf.float32)
    param_trans = tf.Variable(hmr_tran.reshape([1, -1]), dtype=tf.float32)
    initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
    cam_HR = Perspective_Camera(hmr_cam[0], hmr_cam[0], hmr_cam[1],
                                hmr_cam[2], np.zeros(3), np.zeros(3))
    ### get_3d_joints: input 4 * beta
    j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
    j3ds = tf.reshape(j3ds, [-1, 3])
    jointsplus = tf.reshape(jointsplus, [-1, 3])
    v = tf.reshape(v, [-1, 3])
    j2ds_est = []
    verts_est = []
    j2ds_est = cam_HR.project(tf.squeeze(j3ds))
    j2dsplus_est = cam_HR.project(tf.squeeze(jointsplus))
    verts_est = cam_HR.project(tf.squeeze(v))

    HR_mask = tf.convert_to_tensor(HR_masks[ind], dtype=tf.float32)
    verts_est = tf.cast(verts_est, dtype=tf.int64)
    verts_est = tf.concat([tf.expand_dims(verts_est[:, 1], 1),
                           tf.expand_dims(verts_est[:, 0], 1)], 1)

    verts_est_shape = verts_est.get_shape().as_list()
    temp_np = np.ones([verts_est_shape[0]]) * 255
    temp_np = tf.convert_to_tensor(temp_np, dtype=tf.float32)
    delta_shape = tf.convert_to_tensor([HR_masks[ind].shape[0], HR_masks[ind].shape[1]],
                                       dtype=tf.int64)
    scatter = tf.scatter_nd(verts_est, temp_np, delta_shape)
    compare = np.zeros([HR_masks[ind].shape[0], HR_masks[ind].shape[1]])
    compare = tf.convert_to_tensor(compare, dtype=tf.float32)
    scatter = tf.not_equal(scatter, compare)
    scatter = tf.cast(scatter, dtype=tf.float32)
    scatter = scatter * tf.convert_to_tensor([255.0], dtype=tf.float32)

    scatter = tf.expand_dims(scatter, 0)
    scatter = tf.expand_dims(scatter, -1)
    ###########kernel###############
    filter = np.zeros([9, 9, 1])
    filter = tf.convert_to_tensor(filter, dtype=tf.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    scatter = tf.nn.dilation2d(scatter, filter, strides, rates, padding)
    verts2dsilhouette = tf.nn.erosion2d(scatter, filter, strides, rates, padding)
    # tf.gather_nd(verts2dsilhouette, verts_est) = 255
    verts2dsilhouette = tf.squeeze(verts2dsilhouette)
    j2ds_est = tf.convert_to_tensor(j2ds_est)

    objs = {}
    base_weights = np.array(
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  #######
    weights = HR_confs[ind] * base_weights
    weights = tf.constant(weights, dtype=tf.float32)
    objs['J2D_Loss'] = 1.0 * tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_est[2:, :] - HR_j2d), 1))

    base_weights_face = 0.0 * np.array(
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

    objs['mask'] = 0.005 * tf.reduce_sum(100 * verts2dsilhouette / 255.0 * (255.0 - HR_mask) / 255.0
                                        + (255.0 - verts2dsilhouette) / 255.0 * HR_mask / 255.0)

    loss = tf.reduce_mean(objs.values())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # L-BFGS-B
        optimizer = scipy_pt(loss=loss,
                             var_list=[param_trans, param_shape],
                             options={'eps': 1e-12, 'ftol': 1e-12, 'maxiter': 1000, 'disp': False}, method='L-BFGS-B')
        optimizer.minimize(sess)
        cam_HR1 = sess.run([cam_HR.fl_x, cam_HR.cx, cam_HR.cy, cam_HR.trans])
        v_final = sess.run([v, verts_est, j3ds])
        camera = render.camera(cam_HR1[0], cam_HR1[1], cam_HR1[2], cam_HR1[3])
        _, vt = camera.generate_uv(v_final[0], HR_imgs[ind])
        if not os.path.exists(util.hmr_path + "output_shape"):
            os.makedirs(util.hmr_path + "output_shape")
        if util.crop_texture is True:
            img_result_texture, HR_mask_img = camera.render_texture(v_final[0], HR_imgs[ind], vt, HR_masks[ind])
            #if ind == 4:
                #if not os.path.exists(util.texture_path):
                    #os.makedirs(util.texture_path)
                #camera.write_texture_data(util.texture_path, HR_mask_img, vt)
        else:
            img_result_texture, _ = camera.render_texture(v_final[0], HR_imgs[ind], vt)
            #if ind == 4:
                #if not os.path.exists(util.texture_path):
                    #os.makedirs(util.texture_path)
                #camera.write_texture_data(util.texture_path, HR_imgs[ind], vt)
        cv2.imwrite(util.hmr_path + "output_shape/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
        #if util.video is True:
            #videowriter.write(img_result_texture)
        img_result_naked = camera.render_naked(v_final[0], HR_imgs[ind])
        cv2.imwrite(util.hmr_path + "output_shape/hmr_optimization_%04d.png" % ind, img_result_naked)
        img_result_naked_rotation = camera.render_naked_rotation(v_final[0], 90, HR_imgs[ind])
        cv2.imwrite(util.hmr_path + "output_shape/hmr_optimization_rotation_%04d.png" % ind, img_result_naked_rotation)

        # model_f = sess.run(smpl_model.f)
        _objs = sess.run(objs)
        print("the HR j2d loss is %f" % _objs['J2D_Loss'])
        print("the HR J2D_face loss is %f" % _objs['J2D_face_Loss'])
        print("the HR J2D_head loss is %f" % _objs['J2D_head_Loss'])
        print("the HR J2D_foot loss is %f" % _objs['J2D_foot_Loss'])
        print("the HR mask loss is %f" % _objs['mask'])
        # print("the arm_leg_direction loss is %f" % sess.run(objs["arm_leg_direction"]))
        # model_f = model_f.astype(int).tolist()
        #pose_final, betas_final, trans_final = sess.run(
            #[tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])
    # from psbody.meshlite import Mesh
    # m = Mesh(v=np.squeeze(v_final[0]), f=model_f)

    # HR_verts.append(v_final[0])
    #
    # out_ply_path = os.path.join(util.base_path, "HR/output")
    # if not os.path.exists(out_ply_path):
    #     os.makedirs(out_ply_path)
    # out_ply_path = os.path.join(out_ply_path, "%04d.ply" % ind)
    # m.write_ply(out_ply_path)
    #
    #res = {'pose': pose_final, 'betas': betas_final, 'trans': trans_final}
    #with open(util.hmr_path + "output/hmr_optimization_pose_%04d.pkl" % ind, 'wb') as fout:
        #pkl.dump(res, fout)

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