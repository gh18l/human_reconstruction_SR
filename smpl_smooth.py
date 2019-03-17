#coding=utf-8
import glob
import util
import matplotlib.pyplot as plt
from smpl_batch import SMPL
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
import cv2
import os
import numpy as np
import json
import pickle as pkl
import tensorflow as tf
from camera import Perspective_Camera
from opendr_render import render
# TODO
'''
TAKE precedence::fix the position of head
1.add smpl optimization into smooth
warn:2.model is only controlled by cam[t], parameter "trans" doesn't contribute, it need to be corrected.	
'''

def load_openposeCOCO(file):
    #file = "/home/lgh/Documents/2d_3d_con/aa_000000000000_keypoints.json"
    openpose_index = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0, 0, 16, 15, 18, 17, 22, 23, 24, 20, 19, 21])
    deepcut_index = np.array([14, 13, 9, 8, 7, 10, 11, 12, 3, 2, 1, 4, 5, 6]) - 1 #openpose index 123456...

    data = json.load(open(file))
    if len(data["people"]) == 0:
        joints = np.zeros([50, 2])
        conf = np.zeros(50)
        return joints, conf
    _data = data["people"][0]["pose_keypoints_2d"]
    joints = []
    conf = []

    if len(_data) == 45:
        for o in range(0, len(_data), 3):
            temp = []
            temp.append(_data[o])
            temp.append(_data[o + 1])
            joints = np.vstack((joints, temp)) if len(joints) != 0 else temp
            conf = np.vstack((conf, _data[o + 2])) if len(conf) != 0 else np.array([_data[o + 2]])
        conf = conf.reshape((len(conf,)))
        return (joints, conf)

    deepcut_joints = []
    deepcut_conf = []
    for o in range(0, len(_data), 3):   #15 16 17 18 is miss
        temp = []
        temp.append(_data[o])
        temp.append(_data[o + 1])
        joints = np.vstack((joints, temp)) if len(joints) != 0 else temp
        conf = np.vstack((conf, _data[o + 2])) if len(conf) != 0 else np.array([_data[o + 2]])

    #demo_point(joints[:,0], joints[:,1])

    for o in range(14+5+6):
        deepcut_joints = np.vstack((deepcut_joints, joints[openpose_index[o]])) if len(deepcut_joints)!=0 else joints[openpose_index[o]]
        deepcut_conf = np.vstack((deepcut_conf, conf[openpose_index[o]])) if len(deepcut_conf)!=0 else np.array(conf[openpose_index[o]])
    deepcut_conf = deepcut_conf.reshape((len(deepcut_conf)))
    return deepcut_joints, deepcut_conf

def load_deepcut(path):
    est = np.load(path)['pose']
    est = est[:, :, np.newaxis]
    joints = est[:2, :, 0].T
    conf = est[2, :, 0]
    return (joints, conf)

def load_HR_beta():
    pkl_file = os.listdir(HR_path)
    pkl_file = [filename for filename in pkl_file if filename.endswith(".pkl")]
    with open(os.path.join(HR_path, pkl_file[0])) as f:
        param = pkl.load(f)
    beta = np.array(param['betas'])
    beta = beta[np.newaxis, :]
    return beta

def load_pose(path):
    #HR_betas = load_HR_beta()
    imgs = []
    j2ds = []
    confs = []
    j2ds_head = []
    confs_head = []
    j2ds_face = []
    confs_face = []
    j2ds_foot = []
    confs_foot = []
    poses = []
    betas = []
    trans = []
    cams = []
    params = []

    data_path = path + "/optimization_data"
    COCO_path = data_path + "/COCO"
    MPI_path = data_path + "/MPI"
    output_path = path + "/output"

    COCO_j2d_files = os.listdir(COCO_path)
    COCO_j2d_files = sorted([filename for filename in COCO_j2d_files if filename.endswith(".json")],
                            key=lambda d: int((d.split('_')[0])))
    MPI_j2d_files = os.listdir(MPI_path)
    MPI_j2d_files = sorted([filename for filename in MPI_j2d_files if filename.endswith(".json")],
                           key=lambda d: int((d.split('_')[0])))
    img_files = os.listdir(data_path)
    img_files = sorted([filename for filename in img_files if
                        (filename.endswith(".png") or filename.endswith(".jpg")) and "mask" not in filename])
    # key=lambda d: int((d.split('_')[0])))
    pkl_files = os.listdir(output_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                        key=lambda d: int(d.split('_')[3].split('.')[0]))
    #start_ind = int(json_files[0].split('_')[1])
    for ind in range(0, util.BATCH_FRAME_NUM):
        coco_j2d_file_path = os.path.join(COCO_path, COCO_j2d_files[ind])
        coco_j2d, coco_conf = load_openposeCOCO(coco_j2d_file_path)
        mpi_j2d_file_path = os.path.join(MPI_path, MPI_j2d_files[ind])
        mpi_j2d, mpi_conf = load_openposeCOCO(mpi_j2d_file_path)
        j2ds.append(coco_j2d[0:14, :])
        confs.append(coco_conf[0:14])
        j2ds_face.append(coco_j2d[14:19, :])
        confs_face.append(coco_conf[14:19])
        j2ds_head.append(mpi_j2d[1::-1, :])
        confs_head.append(mpi_conf[1::-1])
        j2ds_foot.append(coco_j2d[19:25, :])
        confs_foot.append(coco_conf[19:25])

        ## concatenate
        full_j2d = np.concatenate([coco_j2d[0:14, :], mpi_j2d[1::-1, :]])

        img_file_path = os.path.join(data_path, img_files[ind])
        img = cv2.imread(img_file_path)
        imgs.append(img)
        with open(os.path.join(output_path, pkl_files[ind])) as f:
            param = pkl.load(f)
        params.append(param)

        _pose = np.array(param['pose'])
        #_pose = _pose[np.newaxis, :]
        poses.append(_pose)
        _beta = np.array(param['betas'])
        #_beta = _beta[np.newaxis, :]
        betas.append(_beta)
        _tran = np.array(param['trans'])
        #_tran = _tran[np.newaxis, :]
        trans.append(_tran)
        _cam = np.array(param['cam_LR1'])


        #####下面这个不确定参数的尺度，等之后debug看下
        cams.append(_cam)

        data_dict = {"j2ds": j2ds, "confs": confs, "imgs": imgs,
                     "j2ds_face": j2ds_face,"confs_face": confs_face, "j2ds_head": j2ds_head,
                     "confs_head": confs_head, "j2ds_foot": j2ds_foot,
                     "confs_foot": confs_foot}

    return cams, poses, betas, trans, data_dict

def draw_3D_dot(x, y, z):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b')
    plt.show()

def draw_3D_vertex_joints(x, y, z, x_joints, y_joints, z_joints):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', s = 10)
    for i in range(0, 24):
        ax.scatter(x_joints[i], y_joints[i], z_joints[i], c='r', marker="^", s=200)
    plt.show()

def draw_2D_dot_in_img(img, x, y):
    import matplotlib.pyplot as plt
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.figure(1)
    plt.imshow(img)
    ax = plt.subplot(111)
    ax.scatter(x, y)
    plt.show()

def fix_head_pose(final_pose):
    fix_pose36 = final_pose[0][:, 36:39]
    fix_pose45 = final_pose[0][:, 45:48]
    for i in range(0, len(final_pose)):
        final_pose[i][:, 36:39] = fix_pose36
        final_pose[i][:, 45:48] = fix_pose45
    return final_pose


def main():
    cams, poses, betas, trans, data_dict = load_pose(util.hmr_path)
    LR_j2ds = data_dict["j2ds"]
    LR_confs = data_dict["confs"]
    LR_j2ds_face = data_dict["j2ds_face"]
    LR_confs_face = data_dict["confs_face"]
    LR_j2ds_head = data_dict["j2ds_head"]
    LR_confs_head = data_dict["confs_head"]
    LR_j2ds_foot = data_dict["j2ds_foot"]
    LR_confs_foot = data_dict["confs_foot"]
    LR_imgs = data_dict["imgs"]
    dct_mtx = util.load_dct_base()
    dct_mtx = tf.constant(dct_mtx.T, dtype=tf.float32)

    # For SMPL parameters
    params_tem = []
    params_pose_tem = []
    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)

    # For DCT prior params
    c_dct = tf.Variable(np.zeros([len(util.TEM_SMPL_JOINT_IDS), 3, util.DCT_NUM]), dtype=tf.float32, name='C_DCT')
    _, pose_mean, pose_covariance = util.load_initial_param()
    pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)

    objs = {}
    for idx in range(0, util.BATCH_FRAME_NUM):
        param_pose = tf.Variable(poses[idx], dtype=tf.float32, name='Pose_%d' % idx)
        param_trans = tf.constant(trans[idx], dtype=tf.float32)
        param_shape = tf.constant(betas[idx], dtype=tf.float32)
        param = tf.concat([param_shape, param_pose, param_trans], axis=1)
        params_tem.append(param)
        params_pose_tem.append(param_pose)
        j3ds, vs, jointsplus = smpl_model.get_3d_joints(param, util.TEM_SMPL_JOINT_IDS)  # N x M x 3
        j3ds = tf.reshape(j3ds, [-1, 3])
        jointsplus = tf.reshape(jointsplus, [-1, 3])

        cam = Perspective_Camera(cams[idx][0], cams[idx][0], cams[idx][1],
                                 cams[idx][2], cams[idx][3], np.zeros(3))

        j2ds_est = cam.project(tf.squeeze(j3ds))
        j2dsplus_est = cam.project(tf.squeeze(jointsplus))
        base_weights = 1.0 * np.array(
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        weights = LR_confs[idx] * base_weights
        weights = tf.constant(weights, dtype=tf.float32)
        objs['J2D_Loss_%d' % idx] = tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_est[2:, :] - LR_j2ds[idx]), 1))

        base_weights_face = 0.0 * np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0])
        weights_face = LR_confs_face[idx] * base_weights_face
        weights_face = tf.constant(weights_face, dtype=tf.float32)
        objs['J2D_face_Loss_%d' % idx] = tf.reduce_sum(
            weights_face * tf.reduce_sum(tf.square(j2dsplus_est[14:19, :] - LR_j2ds_face[idx]), 1))

        base_weights_head = 0.0 * np.array(
            [1.0, 1.0])
        weights_head = LR_confs_head[idx] * base_weights_head
        weights_head = tf.constant(weights_head, dtype=tf.float32)
        objs['J2D_head_Loss_%d' % idx] = tf.reduce_sum(
            weights_head * tf.reduce_sum(tf.square(LR_j2ds_head[idx] - j2ds_est[14:16, :]), 1))

        base_weights_foot = 0.0 * np.array(
            [1.0, 1.0])
        _LR_confs_foot = np.zeros(2)
        if LR_confs_foot[idx][0] != 0 and LR_confs_foot[idx][1] != 0:
            _LR_confs_foot[0] = (LR_confs_foot[idx][0] + LR_confs_foot[idx][1]) / 2.0
        else:
            _LR_confs_foot[0] = 0.0
        if LR_confs_foot[idx][3] != 0 and LR_confs_foot[idx][4] != 0:
            _LR_confs_foot[1] = (LR_confs_foot[idx][3] + LR_confs_foot[idx][4]) / 2.0
        else:
            _LR_confs_foot[1] = 0.0
        weights_foot = _LR_confs_foot * base_weights_foot
        weights_foot = tf.constant(weights_foot, dtype=tf.float32)
        _LR_j2ds_foot = np.zeros([2, 2])
        _LR_j2ds_foot[0, 0] = (LR_j2ds_foot[idx][0, 0] + LR_j2ds_foot[idx][1, 0]) / 2.0
        _LR_j2ds_foot[0, 1] = (LR_j2ds_foot[idx][0, 1] + LR_j2ds_foot[idx][1, 1]) / 2.0
        _LR_j2ds_foot[1, 0] = (LR_j2ds_foot[idx][3, 0] + LR_j2ds_foot[idx][4, 0]) / 2.0
        _LR_j2ds_foot[1, 1] = (LR_j2ds_foot[idx][3, 1] + LR_j2ds_foot[idx][4, 1]) / 2.0
        objs['J2D_foot_Loss_%d' % idx] = tf.reduce_sum(
            weights_foot * tf.reduce_sum(tf.square(_LR_j2ds_foot - j2ds_est[0:2, :]), 1))
        pose_diff = param_pose[:, -69:] - pose_mean
        objs['Prior_Loss_%d' % idx] = 5 * tf.squeeze(
            tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))
    params_tem = tf.concat(params_tem, axis=0)
    j3ds_full, vs_full, jointsplus_full = smpl_model.get_3d_joints(params_tem, util.TEM_SMPL_JOINT_IDS)  # N x M x 3
    jointsplus_full = jointsplus_full[:, 14:19]
    for i, jid in enumerate(util.TEM_SMPL_JOINT_IDS):
        for j, aid in enumerate([0, 1, 2]):
            # for j, aid in enumerate([0, 2]):
            trajectory = j3ds_full[:, i, aid]
            '''
            c_dct_initial = tf.matmul(tf.expand_dims(trajectory, axis=0), dct_mtx)
            c_dct_initial = tf.squeeze(c_dct_initial)
            '''

            # import ipdb; ipdb.set_trace()
            # with tf.control_dependencies( [tf.assign(c_dct[i, j], c_dct_initial)] ):
            trajectory_dct = tf.matmul(dct_mtx, tf.expand_dims(c_dct[i, j], axis=-1))
            trajectory_dct = tf.squeeze(trajectory_dct)

            objs['DCT_%d_%d' % (i, j)] = 0.0 * tf.reduce_sum(tf.square(trajectory - trajectory_dct))
    # for i in range(5):
    #     for j, aid in enumerate([0, 1, 2]):
    #         # for j, aid in enumerate([0, 2]):
    #         trajectory = jointsplus_full[:, i, aid]
    #         '''
    #         c_dct_initial = tf.matmul(tf.expand_dims(trajectory, axis=0), dct_mtx)
    #         c_dct_initial = tf.squeeze(c_dct_initial)
    #         '''
    #
    #         # import ipdb; ipdb.set_trace()
    #         # with tf.control_dependencies( [tf.assign(c_dct[i, j], c_dct_initial)] ):
    #         trajectory_dct = tf.matmul(dct_mtx, tf.expand_dims(c_dct[i, j], axis=-1))
    #         trajectory_dct = tf.squeeze(trajectory_dct)
    #
    #         objs['DCT_face_%d_%d' % (i, j)] = 0 * tf.reduce_sum(tf.square(trajectory - trajectory_dct))
    loss = tf.reduce_mean(objs.values())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if util.VIS_OR_NOT:
        func_lc = None
    else:
        func_lc = None

        optimizer = scipy_pt(loss=loss, var_list=params_pose_tem + [c_dct],
                             options={'ftol': 0.001, 'maxiter': 500, 'disp': True}, method='L-BFGS-B')
        # optimizer.minimize(sess, fetches = [objs], loss_callback=func_lc)
        optimizer.minimize(sess, loss_callback=func_lc)

    v_final = sess.run(vs_full)
    pose_final = sess.run(params_pose_tem)

    for fid in range(util.BATCH_FRAME_NUM):
        camera = render.camera(cams[fid][0], cams[fid][1], cams[fid][2], cams[fid][3])
        #texture_img = cv2.resize(texture_img, (util.img_width, util.img_height))

    ### set nonrigid template
    #smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    #template = np.load(util.texture_path + "template.npy")
    # smpl.set_template(template)
    #v = smpl.get_verts(pose_final, betas_final, trans_final)

    #texture_img = cv2.resize(texture_img, (util.img_width, util.img_height))
    #img_result_texture = camera.render_texture(v, texture_img, texture_vt)
    # img_result_texture = tex.correct_render_small(img_result_texture)
        if not os.path.exists(util.hmr_path + "output_dct"):
            os.makedirs(util.hmr_path + "output_dct")
        #cv2.imwrite(util.hmr_path + "output/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
        #img_bg = cv2.resize(LR_imgs[fid], (util.img_width, util.img_height))
        #img_result_texture_bg = camera.render_texture_imgbg(img_result_texture, img_bg)
        #cv2.imwrite(util.hmr_path + "output/texture_bg_%04d.png" % ind,
                #img_result_texture_bg)
    #if util.video is True:
        #videowriter.write(img_result_texture)
        img_result_naked = camera.render_naked(v_final[fid], LR_imgs[fid])
        cv2.imwrite(util.hmr_path + "output_dct/hmr_optimization_%04d.png" % fid, img_result_naked)
        img_result_naked_rotation = camera.render_naked_rotation(v_final[fid], 90, LR_imgs[fid])
        cv2.imwrite(util.hmr_path + "output_dct/hmr_optimization_rotation_%04d.png" % fid, img_result_naked_rotation)


if __name__ == '__main__':
    main()