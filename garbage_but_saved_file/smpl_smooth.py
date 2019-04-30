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

# TODO
'''
TAKE precedence::fix the position of head
1.add smpl optimization into smooth
warn:2.model is only controlled by cam[t], parameter "trans" doesn't contribute, it need to be corrected.	
'''


avi_path = "/home/lgh/code/SMPLify/smplify_public/code/temp/aa1.avi"
json_path = "/home/lgh/code/SMPLify/smplify_public/code/temp"
pkl_path = "/home/lgh/code/SMPLify/smplify_public/code/temp/output"
output_path = "/home/lgh/code/SMPLify/smplify_public/code/temp/DCToutput"
HR_path = "/home/lgh/code/SMPLify/smplify_public/code/temp/HRtemp/output"
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
    COCO_path = path + "/COCO"
    MPI_path = path + "/MPI"
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


        img_file_path = os.path.join(data_path, img_files[ind])
        img = cv2.imread(img_file_path)
        imgs.append(img)
        with open(os.path.join(pkl_path, pkl_files[ind])) as f:
            param = pkl.load(f)
        params.append(param)

        _pose = np.array(param['pose'])
        _pose = _pose[np.newaxis, :]
        poses.append(_pose)
        _beta = np.array(param['betas'])
        _beta = _beta[np.newaxis, :]
        betas.append(_beta)
        _tran = np.array(param['trans'])
        _tran = _tran[np.newaxis, :]
        trans.append(_tran)
        _cam = np.array(param['cam_LR1'])


        #####下面这个不确定参数的尺度，等之后debug看下
        cam = Perspective_Camera(_cam.fl_x, _cam.fl_x, _cam.cx,
                                 _cam.cy, _cam.trans, np.zeros(3))
        cams.append(cam)

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
    j2ds = np.array(data_dict['j2ds']).reshape([-1, 2])
    j2ds_face = np.array(data_dict['j2ds_face']).reshape([-1, 2])
    j2ds_head = np.array(data_dict['j2ds_head']).reshape([-1, 2])
    j2ds_foot = np.array(data_dict['j2ds_foot']).reshape([-1, 2])
    dct_mtx = util.load_dct_base()
    dct_mtx = tf.constant(dct_mtx.T, dtype=tf.float32)

    # For SMPL parameters
    params_tem = []
    params_pose_tem = []
    for idx in range(0, util.BATCH_FRAME_NUM):
        param_pose = tf.Variable(poses[idx], dtype=tf.float32, name='Pose_%d' % idx)
        param_trans = tf.constant(trans[idx], dtype=tf.float32)
        param_shape = tf.constant(betas[idx], dtype=tf.float32)
        param = tf.concat([param_shape, param_pose, param_trans], axis=1)

        params_tem.append(param)
        params_pose_tem.append(param_pose)
    params_tem = tf.concat(params_tem, axis=0)

    # For DCT prior params
    c_dct = tf.Variable(np.zeros([len(util.TEM_SMPL_JOINT_IDS), 3, util.DCT_NUM]), dtype=tf.float32, name='C_DCT')
    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)

    j3ds, vs, jointsplus = smpl_model.get_3d_joints(params_tem, util.TEM_SMPL_JOINT_IDS)  # N x M x 3

    # j3ds, vs = smpl_model.get_3d_joints(params_tem, util.SMPL_JOINT_ALL)
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # j3ds_estimation = sess.run(j3ds)
    # vs_estimation = sess.run(vs)
    # for i in range(0, 100):
    #     draw_3D_vertex_joints(vs_estimation[i, :, 0], vs_estimation[i, :, 1], vs_estimation[i, :, 2],
    #         j3ds_estimation[i, :, 0], j3ds_estimation[i, :, 1], j3ds_estimation[i, :, 2])

    j3ds = j3ds[:, :-1]
    j3ds_flatten = tf.reshape(j3ds, [-1, 3])
    j2ds_est = []

    for idx in range(0, util.BATCH_FRAME_NUM):
        tmp = cams[idx].project(j3ds_flatten[idx * 13 : (idx + 1) * 13, :])
        j2ds_est.append(tmp)
    # for idx in range(0, util.NUM_VIEW):
    #     tmp = cam.project(j3ds_flatten)
    #     j2ds_est.append(tmp)
    j2ds_est = tf.concat(j2ds_est, axis=0)
    j2ds_est = tf.reshape(j2ds_est, [util.NUM_VIEW, util.BATCH_FRAME_NUM, len(util.TEM_SMPL_JOINT_IDS), 2])
    j2ds_est = tf.transpose(j2ds_est, [1, 0, 2, 3])
    j2ds_est = tf.reshape(j2ds_est, [-1, 2])

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # j2ds_estimation = sess.run(j2ds_est)
    # draw_2D_dot_in_img(imgs[0], j2ds_estimation[0:13, 0], j2ds_estimation[0:13, 1])

    _, pose_mean, pose_covariance = util.load_initial_param()
    pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)

    objs = {}
    objs['J2D_Loss'] = tf.reduce_sum(tf.square(j2ds_est - j2ds))
    for i in range(0, util.BATCH_FRAME_NUM):
        pose_diff = params_pose_tem[i][:, -69:] - pose_mean
        objs['Prior_Loss_%d' % i] = 5 * tf.squeeze(
            tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))

    for i, jid in enumerate(util.TEM_SMPL_JOINT_IDS):
        for j, aid in enumerate([0, 1, 2]):
            # for j, aid in enumerate([0, 2]):
            trajectory = j3ds[:, i, aid]
            '''
            c_dct_initial = tf.matmul(tf.expand_dims(trajectory, axis=0), dct_mtx)
            c_dct_initial = tf.squeeze(c_dct_initial)
            '''

            # import ipdb; ipdb.set_trace()
            # with tf.control_dependencies( [tf.assign(c_dct[i, j], c_dct_initial)] ):
            trajectory_dct = tf.matmul(dct_mtx, tf.expand_dims(c_dct[i, j], axis=-1))
            trajectory_dct = tf.squeeze(trajectory_dct)

            objs['DCT_%d_%d' % (i, j)] = 2000 * tf.reduce_sum(tf.square(trajectory - trajectory_dct))
    loss = tf.reduce_mean(objs.values())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def lc(j2d_est):
        _, ax = plt.subplots(1, 3)
        for idx in range(0, util.NUM_VIEW):
            import copy
            tmp = copy.copy(imgs[idx])
            for j2d in j2ds[idx]:
                x = int(j2d[1])
                y = int(j2d[0])

                if x > imgs[0].shape[0] or x > imgs[0].shape[1]:
                    continue
                tmp[x:x + 5, y:y + 5, :] = np.array([0, 0, 255])

            for j2d in j2d_est[idx]:
                x = int(j2d[1])
                y = int(j2d[0])

                if x > imgs[0].shape[0] or x > imgs[0].shape[1]:
                    continue
                tmp[x:x + 5, y:y + 5, :] = np.array([255, 0, 0])
            ax[idx].imshow(tmp)
        plt.show()

    if util.VIS_OR_NOT:
        func_lc = None
    else:
        func_lc = None

        optimizer = scipy_pt(loss=loss, var_list=params_pose_tem + [c_dct],
                             options={'ftol': 0.001, 'maxiter': 500, 'disp': True}, method='L-BFGS-B')
        # optimizer.minimize(sess, fetches = [objs], loss_callback=func_lc)
        optimizer.minimize(sess, loss_callback=func_lc)

    print sess.run(c_dct)

    vs_final = sess.run(vs)
    pose_final = sess.run(params_pose_tem)
    pose_final = fix_head_pose(pose_final)

    betas = sess.run(param_shape)

    model_f = sess.run(smpl_model.f)
    model_f = model_f.astype(int).tolist()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for fid in range(0, util.BATCH_FRAME_NUM):
        from psbody.meshlite import Mesh
        m = Mesh(v=vs_final[fid], f=model_f)
        out_ply_path = "%s/%04d.ply" % (output_path, fid)

        m.write_ply(out_ply_path)

        res = {'pose': pose_final[fid], 'betas': betas, 'trans': trans[fid], 't': params[fid]['t'],
               'rt': params[fid]['rt'], 'f': params[fid]['f']}
        out_pkl_path = out_ply_path.replace('.ply', '.pkl')

        with open(out_pkl_path, 'wb') as fout:
            pkl.dump(res, fout)

if __name__ == '__main__':
    main()