import numpy as np
from smpl_batch import SMPL
import util
from camera import Perspective_Camera
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
import os
import cv2
import pickle as pkl
import hmr
try:
    from smpl.serialization import load_model as _load_model
except:
    from smpl.smpl_webuser.serialization import load_model as _load_model
from opendr_render import render
import time
import pickle
import period_new
import optimization_prepare as opt_pre
import configuration as config
def demo_point(x, y, img_path = None):
    import matplotlib.pyplot as plt
    if img_path != None:
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        plt.figure(1)
        plt.imshow(img)
        ax = plt.subplot(111)
        ax.scatter(x, y)
        #plt.savefig("/home/lgh/code/SMPLify/smplify_public/code/temp/HRtemp/aa1_%04d.png" % o)
        plt.show()
    else:
        plt.figure(1)
        ax = plt.subplot(111)
        ax.scatter(x, y)
        plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png")
        #plt.show()

def demo_point_compare(x1, y1, x2, y2, img_path):
    import matplotlib.pyplot as plt
    if img_path != None:
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        plt.figure(1)
        plt.imshow(img)
        ax = plt.subplot(111)
        ax.scatter(x1, y1, c='r')
        ax.scatter(x2, y2, c='g')
        # plt.savefig("/home/lgh/code/SMPLify/smplify_public/code/temp/HRtemp/aa1_%04d.png" % o)
        plt.show()
    else:
        plt.figure(1)
        ax = plt.subplot(111)
        ax.scatter(x1, y1, c='r')
        ax.scatter(x2, y2, c='g')
        # plt.savefig("/home/lgh/code/SMPLify/smplify_public/code/temp/HRtemp/aa1_%04d.png" % o)
        plt.show()

def tf_unique_2d(x):
    x_shape=x.get_shape() #(3,2)
    x1=tf.tile(x,[1,x_shape[0]]) #[[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2=tf.tile(x,[x_shape[0],1]) #[[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2=tf.reshape(x1,[x_shape[0]*x_shape[0],x_shape[1]])
    x2_2=tf.reshape(x2,[x_shape[0]*x_shape[0],x_shape[1]])
    cond=tf.reduce_all(tf.equal(x1_2,x2_2),axis=1)
    cond=tf.reshape(cond,[x_shape[0],x_shape[0]]) #reshaping cond to match x1_2 & x2_2
    cond_shape=cond.get_shape()
    cond_cast=tf.cast(cond,tf.int32) #convertin condition boolean to int
    cond_zeros=tf.zeros(cond_shape,tf.int32) #replicating condition tensor into all 0's

    #CREATING RANGE TENSOR
    r=tf.range(x_shape[0])
    r=tf.add(tf.tile(r,[x_shape[0]]),1)
    r=tf.reshape(r,[x_shape[0],x_shape[0]])

    #converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape,tf.int32),x_shape[0]+1)
    f2 =tf.ones(cond_shape,tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast,cond_zeros),f1,f2) #if false make it max_index+1 else keep it 1

    #multiply range with new int boolean mask
    r_cond_mul=tf.multiply(r,cond_cast2)
    r_cond_mul2=tf.reduce_min(r_cond_mul,axis=1)
    r_cond_mul3,unique_idx=tf.unique(r_cond_mul2)
    r_cond_mul4=tf.subtract(r_cond_mul3,1)

    #get actual values from unique indexes
    op=tf.gather(x,r_cond_mul4)

    return (op)

def flip_orient(body_orient):
    flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
        cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
    flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
    return flipped_orient

def guess_init(model, focal_length, j2d, init_param):
    cids = np.arange(0, 12)
    # map from LSP to SMPL joints
    j2d_here = j2d[cids]
    #j2d_here[:, 1] = 450.0 - j2d_here[:, 1]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    ##chage it to nontensorflow##
    init_param_tf = tf.constant(init_param, dtype=tf.float32)
    j3ds_tf, vs_tf = model.get_3d_joints(init_param_tf, util.TEM_SMPL_JOINT_IDS)
    with tf.Session() as sess:
        j3ds= sess.run(j3ds_tf)
    # 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
    j3ds = j3ds.squeeze()
    diff3d = np.array([j3ds[9] - j3ds[3], j3ds[8] - j3ds[2]])
    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

    diff2d = np.array([j2d_here[9] - j2d_here[3], j2d_here[8] - j2d_here[2]])
    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))
    # 3D distance - 2D distance
    est_d = focal_length * (mean_height3d / mean_height2d)
    # just set the z value
    init_t = np.array([0., 0., est_d])
    return init_t

def initialize_camera(smpl_model,
                      j2d,
                      img,
                      init_param,
                      flength,   #5000.
                      pix_thsh=25.,
                      viz=False):
    # corresponding SMPL torso ids
    torso_smpl_ids = [2, 1, 17, 16]

    init_camera_t = guess_init(smpl_model, flength, j2d, init_param)
    #init_camera_t = 0. - init_camera_t
    #init_camera_t = np.array([-0.3218652, 0.20879896, -15.358145])

    param_shape = tf.constant(init_param[:10].reshape([1, -1]), dtype=tf.float32)
    param_rot = tf.constant(init_param[10:13].reshape([1, -1]), dtype=tf.float32)
    param_pose = tf.constant(init_param[13:82].reshape([1, -1]), dtype=tf.float32)
    param_trans = tf.constant(init_param[-3:].reshape([1, -1]), dtype=tf.float32)
    param = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)

    camera_t = tf.Variable(init_camera_t, dtype=tf.float32)
    init_camera_r = np.zeros([3])  #######################not sure

    # check how close the shoulder joints are
    try_both_orient = np.linalg.norm(j2d[8] - j2d[9]) < pix_thsh

    # initialize the camera
    #dont know why divide 1000???????????????????????????

    cam = Perspective_Camera(flength, flength, img.shape[1] / 2,
                             img.shape[0] / 2, camera_t, init_camera_r)

    #init_param_tf = tf.constant(init_param, tf.float32)
    j3ds, vs = smpl_model.get_3d_joints(param, util.TEM_SMPL_JOINT_IDS)
    j3ds = tf.reshape(j3ds,[-1, 3])
    j2d_estimation = cam.project(tf.squeeze(j3ds))
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # j2d_estimation1 = sess.run(j2d_estimation)
    # j3ds1 = sess.run(j3ds)
    # demo_point(j2d_estimation1[:, 0], j2d_estimation1[:, 1],
    #            "/home/lgh/code/SMPLify_TF/test/temp/prediction_img_0047_vis.png")
    j2d_estimation = tf.convert_to_tensor(j2d_estimation)
    ######################################################
    ###########################################
    # no optimization with body orientation!!!!!!!!!!!!!!!!!the last three of param_shape
    #the 1e2 is not sure#####
    ##################################################################
    #############################################

    objs = {}
    j2d_flip = j2d.copy()
    for j, jdx in enumerate(util.TORSO_IDS):
        ############convert img coordination into 3D coordination################
        j2d_flip[jdx, 1] = img.shape[0] - j2d[jdx, 1]
        ##########################################################################
        objs["j2d_loss_camera_%d" % j] = tf.reduce_sum(tf.square(j2d_estimation[jdx] - j2d_flip[jdx]))
    #objs["regularization"] = 1e2 * tf.squeeze(tf.reduce_sum(tf.square(init_camera_t - camera_t)))
    loss = tf.reduce_sum(objs.values())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimizer = scipy_pt(loss=loss, var_list=[camera_t],
                     options={'ftol': 0.00000001, 'maxiter': 500, 'disp': False}, method='L-BFGS-B')
        optimizer.minimize(sess)
        final_camera_t = sess.run(camera_t)
        cam.trans = tf.Variable(final_camera_t, dtype=tf.float32)
    return cam, final_camera_t


def output_hmr():
    model = _load_model(util.SMPL_PATH)
    HR_j2ds, HR_confs, HR_imgs, HR_masks = util.load_HR_data()
    LR_j2ds, LR_confs, LR_imgs, LR_masks = util.load_LR_data()
    hmr_thetas, hmr_betas, hmr_trans, hmr_cams = hmr.get_hmr(HR_imgs[0])
    # initial_param, pose_mean, pose_covariance = util.load_initial_param()
    # pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    # pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)

    smpl_model = SMPL(util.SMPL_PATH)
    HR_verts = []
    j3ds_old = []
    ###########################################################
    for ind, _ in enumerate(hmr_thetas):
        hmr_theta = hmr_thetas[ind, :].squeeze()
        hmr_shape = hmr_betas[ind, :].squeeze()
        hmr_tran = hmr_trans[ind, :].squeeze()
        hmr_cam = hmr_cams[ind, :].squeeze()
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
        j3ds, v = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
        j3ds = tf.reshape(j3ds, [-1, 3])
        v = tf.reshape(v, [-1, 3])
        j2ds_est = []
        verts_est = []
        j2ds_est = cam_HR.project(tf.squeeze(j3ds))
        verts_est = cam_HR.project(tf.squeeze(v))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        j2ds_est = sess.run(j2ds_est)
        cam_HR1 = sess.run([cam_HR.fl_x, cam_HR.cx, cam_HR.cy, cam_HR.trans])
        verts_est1 = sess.run(verts_est)
        v1 = sess.run(v)
        j3ds1 = sess.run(j3ds)
        camera = render.camera(cam_HR1[0], cam_HR1[1], cam_HR1[2], cam_HR1[3])
        img_result = camera.render_naked(v1, HR_imgs[ind])
        cv2.imwrite("/home/lgh/code/SMPLify_TF/test/test_hmr_init/dingjianLR/output/hmr_%04d.png" % ind, img_result)

def load_pose_pkl():
    HR_path = util.hmr_path + "output"
    HR_pkl_files = os.listdir(HR_path)
    HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    j3dss = []
    for ind, HR_pkl_file in enumerate(HR_pkl_files):
        HR_pkl_path = os.path.join(HR_path, HR_pkl_file)
        with open(HR_pkl_path) as f:
            param = pickle.load(f)
        j3ds = param['j3ds']
        j3dss.append(j3ds)
    return j3dss

def main(flength=2500.):
    '''
    hmr : initial value
    last pose : no sense
    smooth : j3ds
    '''

    model = _load_model(util.SMPL_PATH)
    body_parsing_idx = opt_pre.load_body_parsing()
    j3dss = load_pose_pkl()
    # dd = pickle.load(open(util.NORMAL_SMPL_PATH))
    # weights = dd['weights']
    # vert_sym_idxs = dd['vert_sym_idxs']
    # v_template = dd['v_template']
    # leg_index = [1, 4, 7, 10, 2, 5, 8, 11]
    # arm_index = [17, 19, 21, 23, 16, 18, 20, 22, 14, 13]
    # #body_index = [0, 3, 6, 9]
    # body_index = [9]
    # head_index = [12, 15]
    # leg_idx = np.zeros(6890)
    # arm_idx = np.zeros(6890)
    # body_idx = np.zeros(6890)
    # head_idx = np.zeros(6890)
    # test_idx = np.zeros(6890)
    # for _, iii in enumerate(leg_index):
    #     length = len(weights[:, iii])
    #     for ii in range(length):
    #         if weights[ii, iii] > 0.6:
    #             leg_idx[ii] = 1
    #             test_idx[ii] = 1
    # for _, iii in enumerate(arm_index):
    #     length = len(weights[:, iii])
    #     for ii in range(length):
    #         if weights[ii, iii] > 0.6:
    #             arm_idx[ii] = 1
    #             test_idx[ii] = 1
    # for _, iii in enumerate(body_index):
    #     length = len(weights[:, iii])
    #     for ii in range(length):
    #         if weights[ii, iii] > 0.6:
    #             body_idx[ii] = 1
    #             test_idx[ii] = 1
    # for _, iii in enumerate(head_index):
    #     length = len(weights[:, iii])
    #     for ii in range(length):
    #         if weights[ii, iii] > 0.6:
    #             head_idx[ii] = 1
    #             test_idx[ii] = 1
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
    videowriter = []
    if util.video == True:
        fps = 10
        size = (HR_imgs[0].shape[1], HR_imgs[0].shape[0])
        video_path = util.hmr_path + "output/texture.mp4"
        videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)

    initial_param, pose_mean, pose_covariance = util.load_initial_param()
    pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)
    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)
    j3ds_old = []
    pose_final_old = []
    pose_final = []
    vt_HRview = []
    verts_body_old = []
    ###########################################################
    for ind, HR_j2d in enumerate(HR_j2ds):
        print("the HR %d iteration" % ind)
        ################################################
        ##############set initial value#################
        ################################################
        hmr_theta = hmr_thetas[ind, :].squeeze()
        hmr_shape = hmr_betas[ind, :].squeeze()
        hmr_tran = hmr_trans[ind, :].squeeze()
        ### fix camera
        hmr_cam = hmr_cams[0, :].squeeze()

        hmr_joint3d = hmr_joint3ds[ind, :, :]
        print(hmr_joint3d[0, 2])
        print(hmr_joint3d[1, 2])
        print(hmr_joint3d[4, 2])
        print(hmr_joint3d[5, 2])
        print(hmr_joint3d[6, 2])
        print(hmr_joint3d[7, 2])
        print(hmr_joint3d[10, 2])
        print(hmr_joint3d[11, 2])
        posepre_joint3d = j3dss[ind]
        # if util.pedestrian_constraint is True:
        #     arm_error = np.fabs((hmr_joint3d[6, 2] + hmr_joint3d[7, 2]) - (hmr_joint3d[10, 2] + hmr_joint3d[11, 2]))
        #     leg_error = np.fabs((hmr_joint3d[0, 2] + hmr_joint3d[1, 2]) - (hmr_joint3d[5, 2] + hmr_joint3d[4, 2]))
        #     ####arm
        #     ####leg
        #     if leg_error > 0.1:
        #         print("leg_error>0.1")
        #         if hmr_joint3d[0, 2] + hmr_joint3d[1, 2] < hmr_joint3d[5, 2] + hmr_joint3d[4, 2]:
        #             hmr_theta[51] = 0.8
        #             hmr_theta[52] = 1e-8
        #             hmr_theta[53] = 1.0
        #             hmr_theta[58] = 1e-8
        #             forward_arm = "left"
        #         else:
        #             hmr_theta[48] = 0.8
        #             hmr_theta[49] = 1e-8
        #             hmr_theta[50] = -1.0
        #             hmr_theta[55] = 1e-8
        #             forward_arm = "right"
        #     #####arm
        #     else:
        #         print("leg_error<=0.1")
        #         if hmr_joint3d[6, 2] + hmr_joint3d[7, 2] < hmr_joint3d[10, 2] + hmr_joint3d[11, 2]:
        #             hmr_theta[48] = 0.8
        #             hmr_theta[49] = 1e-8
        #             hmr_theta[50] = -1.0
        #             hmr_theta[55] = 1e-8
        #             forward_arm = "right"
        #         else:
        #             hmr_theta[51] = 0.8
        #             hmr_theta[52] = 1e-8
        #             hmr_theta[53] = 1.0
        #             hmr_theta[58] = 1e-8
        #             forward_arm = "left"
        #     print(forward_arm)
        if util.pedestrian_constraint is True:
            arm_error = np.fabs((hmr_joint3d[6, 2] + hmr_joint3d[7, 2]) - (hmr_joint3d[10, 2] + hmr_joint3d[11, 2]))
            leg_error = np.fabs((hmr_joint3d[0, 2] + hmr_joint3d[1, 2]) - (hmr_joint3d[5, 2] + hmr_joint3d[4, 2]))
            ####arm
            ####leg
            if leg_error > 0.0:
                print("leg_error>0.1")
                if posepre_joint3d[2, 2] < posepre_joint3d[7, 2]:
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
                print("leg_error<=0.1")
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
        initial_param_np = np.concatenate([hmr_shape.reshape([1, -1]), hmr_theta.reshape([1, -1]), hmr_tran.reshape([1, -1])], axis=1)

        param_shape = tf.Variable(hmr_shape.reshape([1, -1]), dtype=tf.float32)
        param_rot = tf.Variable(hmr_theta[0:3].reshape([1, -1]), dtype=tf.float32)
        param_pose = tf.Variable(hmr_theta[3:72].reshape([1, -1]), dtype=tf.float32)
        param_trans = tf.Variable(hmr_tran.reshape([1, -1]), dtype=tf.float32)

        ###to get hmr 2d verts
        param_shape_fixed = tf.constant(hmr_shape.reshape([1, -1]), dtype=tf.float32)
        hmr_param_rot_fixed = tf.constant(hmr_theta[0:3].reshape([1, -1]), dtype=tf.float32)
        hmr_param_pose_fixed = tf.constant(hmr_theta[3:72].reshape([1, -1]), dtype=tf.float32)
        hmr_param_trans_fixed = tf.constant(hmr_tran.reshape([1, -1]), dtype=tf.float32)

        ####tensorflow array initial_param_tf
        initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
        initial_param_tf_fixed = tf.concat([param_shape_fixed, hmr_param_rot_fixed, hmr_param_pose_fixed, hmr_param_trans_fixed], axis=1)
        #cam_HR, camera_t_final_HR = initialize_camera(smpl_model, HR_j2ds[0], HR_imgs[0], initial_param_np, flength)
        cam_HR = Perspective_Camera(hmr_cam[0], hmr_cam[0], hmr_cam[1],
                                    hmr_cam[2], np.zeros(3), np.zeros(3))
        j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
        _, v_hmr_fixed, __ = smpl_model.get_3d_joints(initial_param_tf_fixed, util.SMPL_JOINT_IDS)
        j3ds = tf.reshape(j3ds, [-1, 3])
        jointsplus = tf.reshape(jointsplus, [-1, 3])
        hmr_joint3d = tf.constant(hmr_joint3d.reshape([-1, 3]), dtype=tf.float32)
        v = tf.reshape(v, [-1, 3])
        v_hmr_fixed= tf.reshape(v_hmr_fixed, [-1, 3])
        j2ds_est = []
        verts_est_mask = []
        j2ds_est = cam_HR.project(tf.squeeze(j3ds))
        j2dsplus_est = cam_HR.project(tf.squeeze(jointsplus))
        verts_est_mask = cam_HR.project(tf.squeeze(v))
        verts_est = cam_HR.project(tf.squeeze(v))
        verts_est_fixed = cam_HR.project(tf.squeeze(v_hmr_fixed))


        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # verts_est1 = sess.run(verts_est)
        # bg = np.zeros_like(HR_masks[ind])
        # for i in range(len(verts_est1)):
        #     x = np.rint(verts_est1[i, 0]).astype('int')
        #     y = np.rint(verts_est1[i, 1]).astype('int')
        #     bg[y, x] = 255
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # smpl_mask = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
        # for i in range(len(verts_est1)):
        #     x = np.rint(verts_est1[i, 0]).astype('int')
        #     y = np.rint(verts_est1[i, 1]).astype('int')
        #     smpl_mask[y, x] = 127
        # contours, hierarchy = cv2.findContours(smpl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.imshow("1", smpl_mask)
        # #cv2.imshow("2", bg)
        # cv2.waitKey()
        # cam_HR1 = sess.run([cam_HR.fl_x, cam_HR.cx, cam_HR.cy, cam_HR.trans])
        # j2ds_est1 = sess.run(j2ds_est)
        # v1 = sess.run(v)
        # j2dsplus_est_fixed1 = sess.run(j2dsplus_est_fixed)
        # j2ds_est_fixed1 = sess.run(j2ds_est_fixed)
        # camera = render.camera(cam_HR1[0], cam_HR1[1], cam_HR1[2], cam_HR1[3])
        # img_result = camera.render_naked(v1, HR_imgs[ind])
        #cv2.imshow("1", img_result)
        #cv2.waitKey()
        #cv2.imwrite(path + "output/hmr_%04d.png" % ind,img_result)
        ######################convert img coordination into 3D coordination#####################
        #HR_j2d[:, 1] = HR_imgs[ind].shape[0] - HR_j2d[:, 1]
        ########################################################################################

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(1)
        # ax = plt.subplot(111)
        # #ax = fig.add_subplot(111, projection='3d')
        # #ax.scatter(v1[:, 0], v1[:, 1], v1[:, 2], c='b', s=1)
        # ax.scatter(j2ds_est1[14, 0], j2ds_est1[14, 1], c='r')
        # #ax.scatter(HR_j2ds_foot[ind][0, 0], HR_j2ds_foot[ind][0, 1], c='b')
        # #hmr_joint3d = hmr_joint3ds[ind,:,:]
        # #ax.scatter(j3ds1[14, 0], j3ds1[14, 1], j3ds1[14, 2], c='r',s=40)
        # #ax.scatter(j3ds1[13, 0], j3ds1[13, 1], j3ds1[13, 2], c='r',s=40)
        # plt.imshow(HR_imgs[ind])
        # plt.show()
        # #plt.imshow(HR_imgs[ind])
        # #ax.scatter(verts_est1[:, 0], verts_est1[:, 1], c='r')
        # #plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png")


        ##############################convert img coordination into 3D coordination###############
        #########################################################################################

        HR_mask = tf.convert_to_tensor(HR_masks[ind], dtype=tf.float32)
        verts_est_mask = tf.cast(verts_est_mask, dtype=tf.int64)
        verts_est_mask = tf.concat([tf.expand_dims(verts_est_mask[:, 1],1),
                               tf.expand_dims(verts_est_mask[:, 0],1)], 1)

        verts_est_shape = verts_est_mask.get_shape().as_list()
        temp_np = np.ones([verts_est_shape[0]]) * 255
        temp_np = tf.convert_to_tensor(temp_np, dtype=tf.float32)
        delta_shape = tf.convert_to_tensor([HR_masks[ind].shape[0], HR_masks[ind].shape[1]],
                                           dtype=tf.int64)
        scatter = tf.scatter_nd(verts_est_mask, temp_np, delta_shape)
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

        base_weights_face = 2.5 * np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0])
        weights_face = HR_confs_face[ind] * base_weights_face
        weights_face = tf.constant(weights_face, dtype=tf.float32)
        objs['J2D_face_Loss'] = tf.reduce_sum(
            weights_face * tf.reduce_sum(tf.square(j2dsplus_est[14:19, :] - HR_j2ds_face[ind]), 1))
        #objs['J2D_face_Loss'] = 10000000.0 * tf.reduce_sum(
              #tf.square(j2dsplus_est[14, :] - HR_j2ds_face[ind][0, :]))

        base_weights_head = 0.5 * np.array(
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
        #w1 = np.array([4.04 * 1e2, 4.04 * 1e2, 57.4 * 10, 57.4 * 10])
        w1 = np.array([1.04 * 2.0, 1.04 * 2.0, 57.4 * 50, 57.4 * 50])
        w1 = tf.constant(w1, dtype=tf.float32)
        objs["angle_elbow_knee"] = 0.03 * tf.reduce_sum(w1 * [
            tf.exp(param_pose[0, 52]), tf.exp(-param_pose[0, 55]),
                tf.exp(-param_pose[0, 9]),tf.exp(-param_pose[0, 12])])

        objs["angle_head"] = 0.0 * tf.reduce_sum(tf.exp(-param_pose[0, 42]))
        ##############################################################
        ###################mask obj###################################
        ##############################################################
        ##0.05
        objs['mask'] = 0.05 * tf.reduce_sum(verts2dsilhouette / 255.0 * (255.0 - HR_mask) / 255.0
                                            + (255.0 - verts2dsilhouette) / 255.0 * HR_mask / 255.0)

        objs['face'] = 0.0 * tf.reduce_sum(tf.square(hmr_joint3d[14:19] - jointsplus[14:19]))

        #objs['face_pose'] = 0.0 * tf.reduce_sum(tf.square(param_pose[0, 0:69] - hmr_theta[3:72])
                                          #+ tf.square(param_rot[0, 0:3] - hmr_theta[0:3]))
        param_pose_full = tf.concat([param_rot, param_pose], axis=1)
        objs['hmr_constraint'] = 6000.0 * tf.reduce_sum(tf.square(tf.squeeze(param_pose_full) - hmr_theta))
        ### 8000.0
        objs['hmr_hands_constraint'] = 100000.0 * tf.reduce_sum(tf.square(tf.squeeze(param_pose_full)[21] - hmr_theta[21])
                                                    + tf.square(tf.squeeze(param_pose_full)[23] - hmr_theta[23])
                                                        + tf.square(tf.squeeze(param_pose_full)[20] - hmr_theta[20])
                                                            + tf.square(tf.squeeze(param_pose_full)[22] - hmr_theta[22]))
        w_temporal = [0.5, 0.5, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0]
        if ind != 0:
            objs['temporal'] = 1000.0 * tf.reduce_sum(
                w_temporal * tf.reduce_sum(tf.square(j3ds - j3ds_old), 1))
            objs['temporal_pose'] = 0.0 * tf.reduce_sum(
                tf.square(pose_final_old[0, 3:72] - param_pose[0, :]))
            objs['temporal_pose_rot'] = 0.0 * tf.reduce_sum(
                tf.square(pose_final_old[0, 0:3] - param_rot[0, :]))

            ##optical flow constraint
            body_idx = np.array(np.hstack([body_parsing_idx[0],
                                           body_parsing_idx[2]])).squeeze()
            body_idx = body_idx.reshape([-1, 1]).astype(np.int64)
            verts_est_body = tf.gather_nd(verts_est, body_idx)
            ### 0.05
            objs['dense_optflow'] = 0.02 * tf.reduce_sum(tf.square(
                verts_est_body - verts_body_old))

        loss = tf.reduce_mean(objs.values())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #L-BFGS-B
            optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_trans, param_pose],
                             options={'eps': 1e-12, 'ftol': 1e-12, 'maxiter': 1000, 'disp': False}, method='L-BFGS-B')
            start_time = time.time()
            optimizer.minimize(sess)
            duration = time.time() - start_time
            print("minimize is %f" % duration)
            start_time = time.time()
            cam_HR1 = sess.run([cam_HR.fl_x, cam_HR.cx, cam_HR.cy, cam_HR.trans])
            if not os.path.exists(util.hmr_path + "output"):
                os.makedirs(util.hmr_path + "output")
            if not os.path.exists(util.hmr_path + "output/camera_after_optimization.npy"):
                np.save(util.hmr_path + "output/camera_after_optimization.npy", cam_HR1)
            duration = time.time() - start_time
            print("cam_HR1 is %f" % duration)
            start_time = time.time()
            v_final = sess.run([v, verts_est, j3ds])
            v_final_fixed = sess.run(verts_est_fixed)
            duration = time.time() - start_time
            print("v_final is %f" % duration)
            camera = render.camera(cam_HR1[0], cam_HR1[1], cam_HR1[2], cam_HR1[3])
            _, vt = camera.generate_uv(v_final[0], HR_imgs[ind])
            if ind == 4:
                vt_HRview = vt
            if not os.path.exists(util.hmr_path + "output"):
                os.makedirs(util.hmr_path + "output")
            if util.crop_texture is True:
                if ind >= 4:
                    img_result_texture = camera.render_texture(v_final[0], HR_imgs[4], vt_HRview)
                    cv2.imwrite(util.hmr_path + "output/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
                    if util.video is True:
                        videowriter.write(img_result_texture)
                if ind == 4:
                    if not os.path.exists(util.texture_path):
                        os.makedirs(util.texture_path)
                    HR_mask_img = camera.save_texture_img(HR_imgs[ind], HR_masks[ind])
                    camera.write_texture_data(util.texture_path, HR_mask_img, vt)
            else:
                if ind >= 4:
                    img_result_texture = camera.render_texture(v_final[0], HR_imgs[4], vt_HRview)
                    cv2.imwrite(util.hmr_path + "output/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
                    if util.video is True:
                        videowriter.write(img_result_texture)
                if ind == 4:
                    if not os.path.exists(util.texture_path):
                        os.makedirs(util.texture_path)
                    camera.write_texture_data(util.texture_path, HR_imgs[ind], vt)
            img_result_naked = camera.render_naked(v_final[0], HR_imgs[ind])
            cv2.imwrite(util.hmr_path + "output/hmr_optimization_%04d.png" % ind, img_result_naked)
            img_result_naked_rotation = camera.render_naked_rotation(v_final[0], 90, HR_imgs[ind])
            cv2.imwrite(util.hmr_path + "output/hmr_optimization_rotation_%04d.png" % ind, img_result_naked_rotation)

            #model_f = sess.run(smpl_model.f)
            start_time = time.time()
            _objs = sess.run(objs)
            duration = time.time() - start_time
            print("_objs is %f" % duration)
            for name in _objs:
                print("the HR %s loss is %f" % (name, _objs[name]))
            #print("the arm_leg_direction loss is %f" % sess.run(objs["arm_leg_direction"]))
            #model_f = model_f.astype(int).tolist()
            start_time = time.time()
            pose_final, betas_final, trans_final = sess.run(
                [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])
            duration = time.time() - start_time
            print("pose_final is %f" % duration)
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
        res = {'pose': pose_final, 'betas': betas_final, 'trans': trans_final, 'cam_HR': cam_HR1,'j3ds': v_final[2]}
        with open(util.hmr_path + "output/hmr_optimization_pose_%04d.pkl" % ind, 'wb') as fout:
            pkl.dump(res, fout)

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
        cv2.imwrite(util.hmr_path + "output_mask/%04d.png" % ind ,HR_masks[ind])
        ###################update initial param###################
        # HR_init_param_shape = betas_final.reshape([1, -1])
        # HR_init_param_rot = (pose_final.squeeze()[0:3]).reshape([1, -1])
        # HR_init_param_pose = (pose_final.squeeze()[3:]).reshape([1, -1])
        # HR_init_param_trans = trans_final.reshape([1, -1])
        # cam_HR_init_trans = camera_t_final_HR
        j3ds_old = v_final[2]
        pose_final_old = pose_final
        #### get body vertex corresponding 2d index
        if ind != len(HR_j2ds) - 1:
            if ind == 0:   ###first is confident in hmr
                body_idx = np.array(body_parsing_idx[0]).squeeze()  ##body part
                head_idx = np.array(body_parsing_idx[1]).squeeze()
                leg_idx = np.array(body_parsing_idx[2]).squeeze()
                verts_body = np.vstack([v_final_fixed[body_idx],
                                        v_final_fixed[leg_idx]])
                verts_body_old = opt_pre.get_dense_correspondence(verts_body,
                                    HR_imgs[ind], HR_imgs[ind + 1])
            else:
                body_idx = np.array(body_parsing_idx[0]).squeeze()  ##body part
                head_idx = np.array(body_parsing_idx[1]).squeeze()
                leg_idx = np.array(body_parsing_idx[2]).squeeze()
                verts_body = np.vstack([v_final[1][body_idx],
                                        v_final[1][leg_idx]])
                verts_body_old = opt_pre.get_dense_correspondence(verts_body,
                                        HR_imgs[ind], HR_imgs[ind + 1])



    #write_obj_and_translation(util.HR_img_base_path + "/aa1small.jpg",
            #util.HR_img_base_path + "/output", util.LR_img_base_path + "/output")
    period_new.save_pkl_to_csv(util.hmr_path + "output")


if __name__ == '__main__':
    main()
