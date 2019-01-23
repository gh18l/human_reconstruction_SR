import numpy as np
from smpl_batch import SMPL
import util
from camera import Perspective_Camera
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
import os
import cv2
import pickle as pkl
import json
from obj_view import write_obj_and_translation

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
        camera_t_final = sess.run(camera_t)
        cam.trans = tf.Variable(camera_t_final, dtype=tf.float32)
    return cam, camera_t_final


def main(flength=2500.):
    HR_j2ds, HR_confs, HR_imgs, HR_masks = util.load_HR_data()
    LR_j2ds, LR_confs, LR_imgs, LR_masks = util.load_LR_data()
    #util.polyfit(HR_j2ds)
    #LR_j2ds = util.dct_LR_2Dpose(LR_j2ds, LR_imgs, 20)
    initial_param, pose_mean, pose_covariance = util.load_initial_param()
    pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)

    HR_init_param_shape = initial_param[:10].reshape([1, -1])
    LR_init_param_shape = initial_param[:10].reshape([1, -1])

    rot = initial_param.squeeze()[10:13]
    flipped_rot = cv2.Rodrigues(rot)[0].dot(
        cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
    flipped_rot = cv2.Rodrigues(flipped_rot)[0].ravel()
    HR_init_param_rot = rot.reshape([1, -1])
    LR_init_param_rot = rot.reshape([1, -1])
    HR_init_param_pose = initial_param[13:82].reshape([1, -1])
    LR_init_param_pose = initial_param[13:82].reshape([1, -1])
    HR_init_param_trans = initial_param[-3:].reshape([1, -1])
    LR_init_param_trans = initial_param[-3:].reshape([1, -1])
    init_param = np.concatenate([HR_init_param_shape, HR_init_param_rot,
                    HR_init_param_pose, HR_init_param_trans], axis=1)
    smpl_model = SMPL(util.SMPL_PATH)
    cam_HR, camera_t_final_HR = initialize_camera(smpl_model, HR_j2ds[0], HR_imgs[0], init_param, flength)
    cam_LR, camera_t_final_LR = initialize_camera(smpl_model, LR_j2ds[0], LR_imgs[0], init_param, flength)
    cam_LR_init_trans = camera_t_final_LR
    # cam_LR = cam_HR
    # camera_t_final_LR = camera_t_final_HR
    HR_verts = []
    ###########################################################
    for ind, HR_j2d in enumerate(HR_j2ds):
        print("the HR %d iteration" % ind)
        ################################################
        ##############set initial value#################
        ################################################
        param_shape = tf.Variable(HR_init_param_shape, dtype=tf.float32)
        param_rot = tf.Variable(HR_init_param_rot, dtype=tf.float32)
        param_pose = tf.Variable(HR_init_param_pose, dtype=tf.float32)
        param_trans = tf.Variable(HR_init_param_trans, dtype=tf.float32)
        param = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)

        j3ds, v = smpl_model.get_3d_joints(param, util.SMPL_JOINT_IDS)
        j3ds = tf.reshape(j3ds, [-1, 3])
        v = tf.reshape(v, [-1, 3])
        j2ds_est = []
        verts_est = []
        j2ds_est = cam_HR.project(tf.squeeze(j3ds))
        verts_est = cam_HR.project(tf.squeeze(v))

        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # j3ds1 = sess.run(j3ds)
        # j2ds_est = sess.run(j2ds_est)
        # v1 = sess.run(v)
        ######################convert img coordination into 3D coordination#####################
        HR_j2d[:, 1] = HR_imgs[ind].shape[0] - HR_j2d[:, 1]
        ########################################################################################


        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(1)
        # ax = plt.subplot(111)
        # #ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(j2ds_est[:, 0], j2ds_est[:, 1], c='b')
        # ax.scatter(HR_j2d[:, 0], HR_j2d[:, 1], c='r')
        # #plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png")
        # plt.show()

        # mask_est = np.zeros((masks[ind].shape[0], masks[ind].shape[1]), dtype="float32")
        # for i in range(len(verts_est)):
        #     mask_est[int(verts_est[i, 1]), int(verts_est[i, 0])] = 255.

        ##############################convert img coordination into 3D coordination###############
        HR_masks[ind] = cv2.flip(HR_masks[ind], 0)
        #########################################################################################

        HR_mask = tf.convert_to_tensor(HR_masks[ind], dtype=tf.float32)
        verts_est = tf.cast(verts_est, dtype=tf.int64)
        verts_est = tf.concat([tf.expand_dims(verts_est[:, 1],1),
                               tf.expand_dims(verts_est[:, 0],1)], 1)

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

        ##################################################################
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # j2ds_est1 = sess.run(j2ds_est)

        objs = {}
        base_weights = np.array(
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        weights = HR_confs[ind] * base_weights
        weights = tf.constant(weights, dtype=tf.float32)

        objs['J2D_Loss'] = tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_est - HR_j2d), 1))

        pose_diff = tf.reshape(param_pose - pose_mean, [1, -1])
        objs['Prior_Loss'] = 8 * tf.squeeze(tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))
        objs['Prior_Shape'] = 5 * tf.reduce_sum(tf.square(param_shape))
        ##############################################################
        ##########control the angle of the elbow and knee#############
        ##############################################################
        w1 = np.array([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78])
        w1 = tf.constant(w1, dtype=tf.float32)
        objs["angle_elbow_knee"] = 0.1 * tf.reduce_sum(w1 * [
            tf.exp(param_pose[0, 52]), tf.exp(-param_pose[0, 55]),
                tf.exp(-param_pose[0, 9]),tf.exp(-param_pose[0, 12])])

        objs["angle_head"] = 100.0 * tf.reduce_sum(tf.exp(-param_pose[0, 42]))
        ##############################################################
        ##########control the direction of the elbow and knee#########
        ######################not successful##########################
        ##############################################################
        # w2 = np.array([0.6 * 1e2, 0.4 * 1e2])
        # w2 = tf.constant(w2, dtype=tf.float32)
        #objs["arm_leg_direction"] = 500.0 * tf.reduce_sum(tf.cast(tf.greater(tf.divide(param_pose[0, 48], param_pose[0, 45]),
                                                             #0.0), dtype=tf.float32))
        #objs["arm_leg_direction"] = tf.reduce_sum(w2 * [tf.exp(param_pose[0, 48] * param_pose[0, 45]),
            #tf.exp(param_pose[0, 0] * param_pose[0, 3])])
        ##############################################################
        ###################mask obj###################################
        ##############################################################
        # dst_type = cv2.cv.CV_DIST_L2 if cv2.__version__[0] == '2' else cv2.DIST_L2
        # dist_i = cv2.distanceTransform(np.uint8(masks[ind]), dst_type, 5) - 1
        # dist_i[dist_i < 0] = 0
        # dist_i[dist_i > 50] = 50
        # dist_i = tf.convert_to_tensor(dist_i)
        # dist_o = cv2.distanceTransform(255 - np.uint8(masks[ind]), dst_type, 5)
        # dist_o[dist_o > 50] = 50
        # dist_o = tf.convert_to_tensor(dist_o)
        # objs['mask'] = 0.01 * tf.reduce_sum(verts_project_est * dist_o * 100.)
        #objs['mask'] = 0.2 * tf.reduce_sum(tf.gather_nd(255.0 - HR_mask, verts_est) / 255.0)
        objs['mask'] = 0.03 * tf.reduce_sum(verts2dsilhouette / 255.0 * (255.0 - HR_mask) / 255.0
                                            + (255.0 - verts2dsilhouette) / 255.0 * HR_mask / 255.0)

        loss = tf.reduce_mean(objs.values())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #L-BFGS-B
            optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_trans, param_pose, param_shape],
                             options={'eps': 1e-12, 'ftol': 1e-10, 'maxiter': 500, 'disp': False}, method='L-BFGS-B')
            optimizer.minimize(sess)
            v_final = sess.run([v, verts_est])
            model_f = sess.run(smpl_model.f)
            print("the HR j2d loss is %f" % sess.run(objs['J2D_Loss']))
            print("the HR prior loss is %f" % sess.run(objs['Prior_Loss']))
            print("the HR Prior_Shape loss is %f" % sess.run(objs['Prior_Shape']))
            print("the HR angle_elbow_knee loss is %f" % sess.run(objs["angle_elbow_knee"]))
            print("the HR angle_head loss is %f" % sess.run(objs["angle_head"]))
            print("the HR mask loss is %f" % sess.run(objs['mask']))
            #print("the arm_leg_direction loss is %f" % sess.run(objs["arm_leg_direction"]))
            model_f = model_f.astype(int).tolist()
            pose_final, betas_final, trans_final = sess.run(
                [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])

        from psbody.meshlite import Mesh
        m = Mesh(v=np.squeeze(v_final[0]), f=model_f)

        HR_verts.append(v_final[0])

        out_ply_path = os.path.join(util.base_path, "HR/output")
        if not os.path.exists(out_ply_path):
            os.makedirs(out_ply_path)
        out_ply_path = os.path.join(out_ply_path, "%04d.ply" % ind)
        m.write_ply(out_ply_path)

        res = {'pose': pose_final, 'betas': betas_final, 'trans': trans_final,
               'f': np.array([flength, flength]), 'rt': np.zeros([3]),
               't': camera_t_final_HR}
        out_pkl_path = out_ply_path.replace('.ply', '.pkl')
        with open(out_pkl_path, 'wb') as fout:
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

        cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp0/1/HR/output_temp/%04d.png" % ind ,HR_masks[ind])
        ###################update initial param###################
        HR_init_param_shape = betas_final.reshape([1, -1])
        HR_init_param_rot = (pose_final.squeeze()[0:3]).reshape([1, -1])
        HR_init_param_pose = (pose_final.squeeze()[3:]).reshape([1, -1])
        HR_init_param_trans = trans_final.reshape([1, -1])







    #############################################################
    LR_verts = []
    bad_flag = 0
    for ind, LR_j2d in enumerate(LR_j2ds):
        print("%d iteration" % ind)
        if LR_masks[ind].ndim == 3:
            bad_flag = 1
            print("we meet a bad mask!!!!!!")
            continue
        ################################################
        ##############set initial value#################
        ################################################
        param_shape = tf.Variable(LR_init_param_shape, dtype=tf.float32)
        param_rot = tf.Variable(LR_init_param_rot, dtype=tf.float32)
        param_pose = tf.Variable(LR_init_param_pose, dtype=tf.float32)
        param_trans = tf.constant(LR_init_param_trans, dtype=tf.float32)
        param = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
        cam_LR.tran = tf.Variable(cam_LR_init_trans, dtype=tf.float32)

        j3ds, v = smpl_model.get_3d_joints(param, util.SMPL_JOINT_IDS)
        j3ds = tf.reshape(j3ds, [-1, 3])
        v = tf.reshape(v, [-1, 3])

        j2ds_est = []
        verts_est = []
        j2ds_est = cam_LR.project(tf.squeeze(j3ds))
        verts_est = cam_LR.project(tf.squeeze(v))


        ##############################convert img coordination into 3D coordination###############
        LR_masks[ind] = cv2.flip(LR_masks[ind], 0)
        #########################################################################################

        ######################convert img coordination into 3D coordination#####################
        LR_j2d[:, 1] = LR_imgs[ind].shape[0] - LR_j2d[:, 1]
        ########################################################################################

        # mask_est = np.zeros((masks[ind].shape[0], masks[ind].shape[1]), dtype="float32")
        # for i in range(len(verts_est)):
        #     mask_est[int(verts_est[i, 1]), int(verts_est[i, 0])] = 255.
        LR_mask = tf.convert_to_tensor(LR_masks[ind], dtype=tf.float32)
        verts_est = tf.cast(verts_est, dtype=tf.int64)
        verts_est = tf.concat([tf.expand_dims(verts_est[:, 1],1),
                               tf.expand_dims(verts_est[:, 0],1)], 1)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # j2ds_est1 = sess.run(j2ds_est)
        # mask_est1 = sess.run(unique_cantor)
        #verts_est_unique = tf.unique(verts_est)
        ############erode and pengzhang############
        # verts2dsilhouette = np.zeros([LR_masks[ind].shape[0], LR_masks[ind].shape[1]])
        # verts2dsilhouette = tf.convert_to_tensor(verts2dsilhouette, dtype=tf.float32)
        verts_est_shape = verts_est.get_shape().as_list()
        temp_np = np.ones([verts_est_shape[0]]) * 255
        temp_np = tf.convert_to_tensor(temp_np, dtype=tf.float32)
        delta_shape = tf.convert_to_tensor([LR_masks[ind].shape[0], LR_masks[ind].shape[1]],
                                           dtype=tf.int64)
        #delta = tf.SparseTensor(verts_est, temp_np, delta_shape)
        #delta_dense = tf.sparse_to_dense(verts_est, delta_shape, 255.0, 0.0)
        #verts2dsilhouette = verts2dsilhouette + delta_dense
        scatter = tf.scatter_nd(verts_est, temp_np, delta_shape)
        compare = np.zeros([LR_masks[ind].shape[0], LR_masks[ind].shape[1]])
        compare = tf.convert_to_tensor(compare, dtype=tf.float32)
        scatter = tf.not_equal(scatter, compare)
        scatter = tf.cast(scatter, dtype=tf.float32)
        scatter = scatter * tf.convert_to_tensor([255.0], dtype=tf.float32)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # scatter = sess.run(scatter)
        scatter = tf.expand_dims(scatter, 0)
        scatter = tf.expand_dims(scatter, -1)
        ###########kernel###############
        filter = np.zeros([9,9,1])
        filter = tf.convert_to_tensor(filter, dtype=tf.float32)
        strides = [1,1,1,1]
        rates = [1,1,1,1]
        padding = "SAME"
        scatter = tf.nn.dilation2d(scatter, filter, strides, rates, padding)
        verts2dsilhouette = tf.nn.erosion2d(scatter, filter, strides, rates, padding)
        #tf.gather_nd(verts2dsilhouette, verts_est) = 255
        verts2dsilhouette = tf.squeeze(verts2dsilhouette)
        #a = verts_est[:, 1]
        #cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp/est.png", mask_est)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # mask_est = sess.run(verts2dsilhouette)
        # verts_est = sess.run(verts_est)
        # mask_est = mask_est.squeeze()
        # mask_est.astype(np.int8)
        # mask_est[verts_est[:,0], verts_est[:,1]] = 0
        # cv2.imshow("1", mask_est)
        # cv2.waitKey()
        # #demo_point(mask_est.squeeze()[:, 0], mask_est.squeeze()[:, 1])
        # img = np.zeros([LR_masks[ind].shape[1], LR_masks[ind].shape[0]])
        # img[mask_est[:,1], mask_est[:,0]] = 255
        # cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png", img)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # verts_project_est = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # #verts_project_est = tf.convert_to_tensor(verts_project_est, dtype=tf.float32)
        #
        # cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/est1.png", verts_project_est)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # verts_est1 = sess.run(verts_est)
        # j3ds1 = sess.run(j3ds)
        # v1 = sess.run(v)
        # index = [1]
        # a = j2d_estimation1[index]
        # demo_point(v1.squeeze()[:, 0], v1.squeeze()[:, 1])

        j2ds_est = tf.convert_to_tensor(j2ds_est)

        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # a = np.array([[0, 6],[0, 8]])
        # param_pose1 = sess.run(tf.gather_nd(param_pose, a[0]))


        objs = {}
        base_weights = np.array(
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        weights = LR_confs[ind] * base_weights
        weights = tf.constant(weights, dtype=tf.float32)

        objs['J2D_Loss'] = tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_est - LR_j2d), 1))

        pose_diff = tf.reshape(param_pose - pose_mean, [1, -1])
        ######20
        objs['Prior_Loss'] = 10 * tf.squeeze(tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))
        objs['Prior_Shape'] = 5 * tf.reduce_sum(tf.square(param_shape))
        ##############################################################
        ##########control the angle of the elbow and knee#############
        ##############################################################
        w1 = np.array([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78])
        w1 = tf.constant(w1, dtype=tf.float32)
        objs["angle_elbow_knee"] = 0.1 * tf.reduce_sum(w1 * [
            tf.exp(param_pose[0, 52]), tf.exp(-param_pose[0, 55]),
                tf.exp(-param_pose[0, 9]),tf.exp(-param_pose[0, 12])])

        objs["angle_head"] = 100.0 * tf.reduce_sum(tf.exp(-param_pose[0, 42]))
        ##############################################################
        ##########control the direction of the elbow and knee#########
        ######################not successful##########################
        ##############################################################
        w2 = np.array([0.6 * 1e2, 0.4 * 1e2])
        w2 = tf.constant(w2, dtype=tf.float32)
        # objs["arm_leg_direction"] = tf.reduce_sum(w2 * [tf.exp(param_pose[0, 48] * param_pose[0, 45]),
        #     tf.exp(param_pose[0, 0] * param_pose[0, 3])])
        ##############################################################
        ###################mask obj###################################
        ##############################################################
        # dst_type = cv2.cv.CV_DIST_L2 if cv2.__version__[0] == '2' else cv2.DIST_L2
        # dist_i = cv2.distanceTransform(np.uint8(masks[ind]), dst_type, 5) - 1
        # dist_i[dist_i < 0] = 0
        # dist_i[dist_i > 50] = 50
        # dist_i = tf.convert_to_tensor(dist_i)
        # dist_o = cv2.distanceTransform(255 - np.uint8(masks[ind]), dst_type, 5)
        # dist_o[dist_o > 50] = 50
        # dist_o = tf.convert_to_tensor(dist_o)
        # objs['mask'] = 0.01 * tf.reduce_sum(verts_project_est * dist_o * 100.)
        #objs['mask'] = 0.2 * tf.reduce_sum(tf.gather_nd(255.0 - LR_mask, verts_est) / 255.0)
        objs['mask'] = 0.02 * tf.reduce_sum(verts2dsilhouette / 255.0 * (255.0 - LR_mask) / 255.0
                       + (255.0 - verts2dsilhouette) / 255.0 * LR_mask / 255.0)
        loss = tf.reduce_mean(objs.values())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #L-BFGS-B
            optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_pose, param_shape, cam_LR.trans],
                             options={'eps': 1e-12, 'ftol': 1e-10, 'maxiter': 500, 'disp': False}, method='L-BFGS-B')
            optimizer.minimize(sess)
            if ind == 3:
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                j3ds1 = sess.run(j3ds)
                j2ds_est = sess.run(j2ds_est)
                v1 = sess.run(v)

                transs = sess.run(cam_LR.tran)
                camera_mtx = np.array([[2500.0, 0., 300.0], [0., 2500.0, 225.0], [0., 0., 1.]], dtype=np.float64)
                k = np.zeros(5)
                rt = np.array([0.0, 0.0, 0.0])
                t = transs
                uv = cv2.projectPoints(j3ds1, rt, t, camera_mtx, k)[0].squeeze()

                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(1)
                ax = plt.subplot(111)
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(j3ds1[:, 0], j3ds1[:, 1],j3ds1[:, 2], c='b')
                #ax.scatter(LR_j2d[:, 0], LR_j2d[:, 1], c='r')
                # plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png")
                plt.show()
                a = 1

            v_final = sess.run([v, verts_est])
            cam_LR_final = sess.run(cam_LR.trans)
            # if ind == 2:
            #     threedjoints = sess.run(j3ds)
            #     twodjoints = sess.run(j2ds_est)
            #     a = 1
            model_f = sess.run(smpl_model.f)
            print("the LR j2d loss is %f" % sess.run(objs['J2D_Loss']))
            print("the LR prior loss is %f" % sess.run(objs['Prior_Loss']))
            print("the LR Prior_Shape loss is %f" % sess.run(objs['Prior_Shape']))
            print("the LR angle_elbow_knee loss is %f" % sess.run(objs["angle_elbow_knee"]))
            print("the LR mask loss is %f" % sess.run(objs['mask']))
            print("the LR angle_head loss is %f" % sess.run(objs["angle_head"]))
            #print("the arm_leg_direction loss is %f" % sess.run(objs["arm_leg_direction"]))
            model_f = model_f.astype(int).tolist()
            pose_final, betas_final, trans_final = sess.run(
                [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])

        from psbody.meshlite import Mesh
        m = Mesh(v=np.squeeze(v_final[0]), f=model_f)
        LR_verts.append(v_final[0])
        out_ply_path = os.path.join(util.base_path, "LR/output")
        if not os.path.exists(out_ply_path):
            os.makedirs(out_ply_path)
        out_ply_path = os.path.join(out_ply_path, "%04d.ply" % (ind+util.img_first_index+
                                    util.first_index))
        m.write_ply(out_ply_path)

        ##################change direction to render in opengl############
        pose_final_in_opengl = pose_final.copy()
        # pose_final_in_opengl_rot = pose_final_in_opengl.squeeze()[:3]
        # pose_final_in_opengl_rot_flip = cv2.Rodrigues(pose_final_in_opengl_rot)[0].dot(
        #     cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
        # pose_final_in_opengl_rot_flip = cv2.Rodrigues(pose_final_in_opengl_rot_flip)[0].ravel()
        # pose_final_in_opengl[:,:3] = pose_final_in_opengl_rot_flip
        res = {'pose': pose_final, 'betas': betas_final, 'trans': trans_final,
               'f': np.array([flength, flength]), 'rt': np.zeros([3]),
               't': cam_LR_final}
        out_pkl_path = out_ply_path.replace('.ply', '.pkl')
        with open(out_pkl_path, 'wb') as fout:
            pkl.dump(res, fout)

        #####################write 2dverts########################
        verts2d = v_final[1]
        for z in range(len(verts2d)):
            if int(verts2d[z][0]) > LR_masks[ind].shape[0] - 1:
                print(int(verts2d[z][0]))
                verts2d[z][0] = LR_masks[ind].shape[0] - 1
            if int(verts2d[z][1]) > LR_masks[ind].shape[1] - 1:
                print(int(verts2d[z][1]))
                verts2d[z][1] = LR_masks[ind].shape[1] - 1
            (LR_masks[ind])[int(verts2d[z][0]), int(verts2d[z][1])] = 127

        cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/output_temp/%04d.png" % (ind+util.img_first_index+
                                    util.first_index), LR_masks[ind])

        ###################update initial param###################
        if bad_flag == 0:
            LR_init_param_shape = betas_final.reshape([1, -1])
            LR_init_param_rot = (pose_final.squeeze()[0:3]).reshape([1, -1])
            LR_init_param_pose = (pose_final.squeeze()[3:]).reshape([1, -1])
            LR_init_param_trans = trans_final.reshape([1, -1])
            cam_LR_init_trans = cam_LR_final
        else:
            LR_init_param_shape = initial_param[:10].reshape([1, -1])
            LR_init_param_rot = rot.reshape([1, -1])
            LR_init_param_pose = initial_param[13:82].reshape([1, -1])
            LR_init_param_trans = initial_param[-3:].reshape([1, -1])
            cam_LR_init_trans = camera_t_final_LR
            bad_flag = 0

    write_obj_and_translation(util.HR_img_base_path + "/aa1small.jpg",
            util.HR_img_base_path + "/output", util.LR_img_base_path + "/output")



if __name__ == '__main__':
    main()
