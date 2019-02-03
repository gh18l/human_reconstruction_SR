#coding=utf-8
import os
import pickle as pkl
import scipy.io as sio
import numpy as np
from camera import Perspective_Camera
import cv2
import tensorflow as tf
import hmr
import json

GENDER = 'm'
HEVA_PATH = 'Data/HEVA_Validate'
#SMPL_PATH = 'Data/Smpl_Model/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % GENDER
SMPL_PATH = 'Data/Smpl_Model/neutral_smpl_with_cocoplus_reg.pkl'
NORMAL_SMPL_PATH = '/home/lgh/code/SMPLify_TF/Data/Smpl_Model/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
N_BETAS = 10
SMPL_JOINT_IDS = [11, 10, 8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12]
SMPL_JOINT_ALL = range(0, 24)
TORSO_IDS = [3-1, 4-1, 9-1, 10-1]
HEAD_VID = 411
# TODO
NUM_VIEW = 1
# TODO
VIS_OR_NOT = False
# TODO
LOG_OR_NOT = 1
# TODO
BATCH_FRAME_NUM = 30   ###################
# TODO
DCT_NUM = 10
DCT_MAT_PATH = 'Data/DCT_Basis/%d.mat' % BATCH_FRAME_NUM
# TODO
#tem_j2d_ids = [0, 1, 4, 5, 6, 7, 10, 11]
#tem_smpl_joint_ids = [8, 5, 4, 7, 21, 19, 18, 20]
TEM_J2D_IDS = range(0, 13)
TEM_SMPL_JOINT_IDS = SMPL_JOINT_IDS
# TODO
POSE_PRIOR_PATH = 'Data/Prior/genericPrior.mat'

base_path = "/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/"
HR_j2d_base_path = base_path + "/HR"
HR_img_base_path = base_path + "/HR"
HR_mask_base_path = base_path + "/HR"
LR_j2d_base_path = base_path + "/LR"
LR_img_base_path = base_path + "/LR"
LR_mask_base_path = base_path + "/LR"
LR_pkl_base_path = base_path + "/LR/output"
HR_pkl_base_path = base_path + "/HR/output"

LR_j2d_dctsmooth_base_path = base_path + "/LRdctsmooth"
LR_img_dctsmooth_base_path = base_path + "/LRdctsmooth"

hmr_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/"
texture_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/output/texture_file/"
HR_pose_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/output/"
crop_texture = True  ###only use in small texture
index_data = 0
video = True

###dingjianLR100
#lr_points = [0, 16, 31, 47, 64, 80, 96]    ###[0, 18, 36, 54, 72]
#hr_points = [4, 20, 36]
###xiongfei
# lr_points = [0, 18, 36, 54, 72]    ###[0, 18, 36, 54, 72]
# hr_points = [1, 19, 37]
# lr_points = [0, 18]    ###[0, 18, 36, 54, 72]
# hr_points = [1, 19]
###jianing
# lr_points = [0, 16, 32, 48, 64, 80]    ###[0, 18, 36, 54, 72]
# hr_points = [11, 27, 43, 59]
###jianing2
# lr_points = [0, 16, 32, 48, 64, 81]    ###[0, 18, 36, 54, 72]
# hr_points = [14, 30, 46, 62]
###jianing2copy
lr_points = [0, 16]    ###[0, 18, 36, 54, 72]
hr_points = [14, 30]


which_people = "tianyi_LR"
code_params = {"tianyi_LR": {"Prior_Loss":10.0, "Prior_Shape":5.0, "angle_elbow_knee":0.1, "angle_head":100.0,
                        "rot_direction":0.0, "arm_direction":0.0, "mask":0.06, "temporal": 800.0},
               ####################good effect##################
               "dingjian_LR":{"Prior_Loss": 10.0, "Prior_Shape":5.0, "angle_elbow_knee":0.1, "angle_head":100.0,
                        "rot_direction":0.0, "arm_direction":0.0, "mask":0.08, "temporal": 800.0}}
               #"dingjian_LR":{"Prior_Loss": 10.0, "Prior_Shape":5.0, "angle_elbow_knee":0.1, "angle_head":200.0,
                        #"rot_direction":0.0, "arm_direction":0.0, "mask":0.002, "temporal": 800.0}}
               # "dingjian_LR":{"Prior_Loss": 5.0, "Prior_Shape":5.0, "angle_elbow_knee":0.1, "angle_head":250.0,
               #          "rot_direction":200.0, "arm_direction":500.0, "mask":0.05}}

########################index tianyi############################
if index_data == 0:
    img_first_index = 0
    first_index = 11   #242　　　#0   122   160 is wrong  190 is wrong
    last_index = 100    #360   #360
    bad_mask = [489 - img_first_index, 490 - img_first_index, 491 - img_first_index]

########################index 1(dingjian LR)############################
if index_data == 1:
    img_first_index = 80
    first_index = 210   #242　　　#0   122   160 is wrong  190 is wrong
    last_index = 260    #360   #360
    bad_mask = [84 - img_first_index, 104 - img_first_index, 130 - img_first_index,
                170 - img_first_index, 181 - img_first_index, 182 - img_first_index,
                188 - img_first_index, 251 - img_first_index, 252 - img_first_index,
                253 - img_first_index, 310 - img_first_index, 311 - img_first_index,
                312 - img_first_index, 313 - img_first_index]

########################index 2(xiongfei LR)###########################
if index_data == 2:
    img_first_index = 0
    first_index = 0   #242　　　#0
    last_index = 121    #360   #360   121
    bad_mask = [22 - img_first_index, 23 - img_first_index, 187 - img_first_index,
                191 - img_first_index, 201 - img_first_index,
                202 - img_first_index, 203 - img_first_index, 213 - img_first_index,
                214 - img_first_index, 215 - img_first_index, 216 - img_first_index,
                334 - img_first_index, 335 - img_first_index, 311 - img_first_index,
                312 - img_first_index, 313 - img_first_index]


def load_deepcut(path):
    est = np.load(path)['pose']
    est = est[:, :, np.newaxis]
    joints = est[:2, :, 0].T
    conf = est[2, :, 0]
    return (joints, conf)

def load_openposeCOCO(file):
    #file = "/home/lgh/Documents/2d_3d_con/aa_000000000000_keypoints.json"
    openpose_index = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0, 0, 16, 15, 18, 17, 22, 23, 24, 20, 19, 21])
    deepcut_index = np.array([14, 13, 9, 8, 7, 10, 11, 12, 3, 2, 1, 4, 5, 6]) - 1 #openpose index 123456...

    data = json.load(open(file))
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

def load_hmr_data(path):
    hmr_theta, hmr_beta, hmr_tran, hmr_cam, hmr_joint3d = hmr.get_hmr(path)
    hmr_dict = {"hmr_thetas" : hmr_theta, "hmr_betas" : hmr_beta,
                "hmr_trans": hmr_tran, "hmr_cams" : hmr_cam,
                "hmr_joint3ds" : hmr_joint3d}
    path = path + "/optimization_data"
    COCO_path = path + "/COCO"
    MPI_path = path + "/MPI"
    imgs = []
    j2ds = []
    confs = []
    j2ds_head = []
    confs_head = []
    j2ds_face = []
    confs_face = []
    j2ds_foot = []
    confs_foot = []
    masks = []
    COCO_j2d_files = os.listdir(COCO_path)
    COCO_j2d_files = sorted([filename for filename in COCO_j2d_files if filename.endswith(".json")],
                       key=lambda d: int((d.split('_')[0])))
    MPI_j2d_files = os.listdir(MPI_path)
    MPI_j2d_files = sorted([filename for filename in MPI_j2d_files if filename.endswith(".json")],
                            key=lambda d: int((d.split('_')[0])))
    img_files = os.listdir(path)
    img_files = sorted([filename for filename in img_files if filename.endswith(".jpg") and "mask" not in filename])
    mask_files = os.listdir(path)
    mask_files = sorted([filename for filename in mask_files if filename.endswith(".png") and "mask" in filename],
                        key=lambda d: int((d.split('_')[1]).split('.')[0]))

    for ind, COCO_j2d_file in enumerate(COCO_j2d_files):
        coco_j2d_file_path = os.path.join(COCO_path, COCO_j2d_file)
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


        img_file_path = os.path.join(path, img_files[ind])
        img = cv2.imread(img_file_path)
        imgs.append(img)

        mask_file_path = os.path.join(path, mask_files[ind])
        mask1 = cv2.imread(mask_file_path)
        ##################cautious change!!!!!!!!!!!################
        mask = mask1[:, :, 0]
        mask[mask<255]=0
        masks.append(mask)
    data_dict = {"j2ds": j2ds, "confs": confs, "imgs": imgs,
                 "masks": masks, "j2ds_face": j2ds_face,
                 "confs_face": confs_face, "j2ds_head": j2ds_head,
                 "confs_head": confs_head, "j2ds_foot": j2ds_foot,
                 "confs_foot": confs_foot}
    return hmr_dict, data_dict


def load_initial_param():
    pose_prior = sio.loadmat(POSE_PRIOR_PATH, squeeze_me=True, struct_as_record=False)
    pose_mean = pose_prior['mean']
    pose_covariance = np.linalg.inv(pose_prior['covariance'])
    zero_shape = np.ones([13]) * 1e-8 # extra 3 for zero global rotation
    zero_trans = np.ones([3]) * 1e-8

    # pose_mean[45] = np.abs(pose_mean[45]) + 1.5
    # pose_mean[48] = np.abs(pose_mean[48]) + 1.5
    #
    # pose_mean[45] = 1.2
    # pose_mean[46] = 1e-8
    # pose_mean[47] = -1.3
    # # pose_mean[48] = 1.2
    # # pose_mean[49] = 1e-8
    # # pose_mean[50] = 1.3
    # pose_mean[52] = 1e-8
    # pose_mean[55] = 1e-8

    initial_param = np.concatenate([zero_shape, pose_mean, zero_trans], axis=0)

    return initial_param, pose_mean, pose_covariance

def left_hand_to_right_hand(input):
    _input = tf.squeeze(input)
    #input_shape = input.get_shape().as_list()
    coe = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    coe_tf = tf.convert_to_tensor(coe, dtype=tf.float32)
    result = tf.matmul(_input, coe_tf)
    result = tf.expand_dims(result, 0)
    return result
