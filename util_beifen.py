#coding=utf-8
import os
import pickle as pkl
import scipy.io as sio
import numpy as np
from camera import Perspective_Camera
import cv2
import tensorflow as tf

# TODO: choose gender
GENDER = 'm'
HEVA_PATH = 'Data/HEVA_Validate'
SMPL_PATH = 'Data/Smpl_Model/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % GENDER
# TODO: set beta dimension
N_BETAS = 10
SMPL_JOINT_IDS = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12]
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

index_data = 0

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

def load_HR_data():
    HR_imgs = []
    HR_j2ds = []
    confs = []
    masks = []
    j2d_files = os.listdir(HR_j2d_base_path)
    j2d_files = sorted([filename for filename in j2d_files if filename.endswith(".npz")],
						key=lambda d: int((d.split('_')[2]).split('.')[0]))
    img_files = os.listdir(HR_img_base_path)
    img_files = sorted([filename for filename in img_files if filename.endswith(".png") and "img" in filename])
    mask_files = os.listdir(HR_mask_base_path)
    mask_files = sorted([filename for filename in mask_files if filename.endswith(".png") and "mask" in filename],
                       key=lambda d: int((d.split('_')[1]).split('.')[0]))

    for ind, j2d_file in enumerate(j2d_files):
        j2d_file_path = os.path.join(HR_j2d_base_path, j2d_file)
        HR_j2d, conf = load_deepcut(j2d_file_path)
        HR_j2ds.append(HR_j2d)
        confs.append(conf)
        img_file_path = os.path.join(HR_img_base_path, img_files[ind])
        HR_img = cv2.imread(img_file_path)
        HR_imgs.append(HR_img)

        mask_file_path = os.path.join(HR_mask_base_path, mask_files[ind])
        mask1 = cv2.imread(mask_file_path)
        mask = mask1[:, :, 0]
        masks.append(mask)
    return HR_j2ds, confs, HR_imgs, masks

def polyfit(HR_j2ds):
    import matplotlib.pyplot as plt
    length = len(HR_j2ds)
    array = np.zeros((length, 28))
    for ind, HR_j2d in enumerate(HR_j2ds):
        for i in range(14):
            array[ind, i*2] = HR_j2d[i, 0]
            array[ind, i*2+1] = HR_j2d[i, 1]
    for i in range(28):
        x = np.arange(1, length + 1, 1)
        y = array[:, i]
        reg = np.polyfit(x, y, 5)
        ry = np.polyval(reg, x)
        plt.plot(x, y, 'b^', label='f(x)')
        plt.plot(x, ry, 'r.', label='regression')
        plt.legend(loc=0)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()
        a = 1

def load_LR_data():
    LR_imgs = []
    LR_j2ds = []
    confs = []
    masks = []
    j2d_files = os.listdir(LR_j2d_base_path)
    j2d_files = sorted([filename for filename in j2d_files if filename.endswith(".npz")],
						key=lambda d: int((d.split('_')[2]).split('.')[0]))
    img_files = os.listdir(LR_img_base_path)
    img_files = sorted([filename for filename in img_files if filename.endswith(".jpg")])
    mask_files = os.listdir(LR_mask_base_path)
    mask_files = sorted([filename for filename in mask_files if filename.endswith(".png") and "mask" in filename],
                       key=lambda d: int((d.split('_')[1]).split('.')[0]))

    for ind, j2d_file in enumerate(j2d_files[first_index:last_index]):
        j2d_file_path = os.path.join(LR_j2d_base_path, j2d_file)
        LR_j2d, conf = load_deepcut(j2d_file_path)
        LR_j2ds.append(LR_j2d)
        confs.append(conf)

        img_file_path = os.path.join(LR_img_base_path, img_files[ind + first_index])
        LR_img = cv2.imread(img_file_path)
        LR_imgs.append(LR_img)

        mask_file_path = os.path.join(LR_mask_base_path, mask_files[ind + first_index])
        mask1 = cv2.imread(mask_file_path)
        ##################cautious change!!!!!!!!!!!################
        if ind + first_index in bad_mask:
            mask = np.zeros([100, 100, 3])
        else:
            mask = mask1[:, :, 0]
        masks.append(mask)
    return LR_j2ds, confs, LR_imgs, masks

def dct_LR_2Dpose(joints, imgs, dct_coe):
    j2d_files = os.listdir(LR_j2d_dctsmooth_base_path)
    j2d_files = sorted([filename for filename in j2d_files if filename.endswith(".npz")],
                       key=lambda d: int((d.split('_')[2]).split('.')[0]))
    img_files = os.listdir(LR_img_dctsmooth_base_path)
    img_files = sorted([filename for filename in img_files if filename.endswith(".png")])
    length = len(j2d_files)
    array = np.zeros((length, 28))
    fre = np.zeros((length))
    ifre = np.zeros((length))
    new_joints = np.zeros((length, 14, 2))
    new_conf = np.zeros((length, 14))
    for ind, LR_j2d in enumerate(joints):
        #j2d_file_path = os.path.join(LR_j2d_dctsmooth_base_path, j2d_file)
        #LR_j2d, conf = load_deepcut(j2d_file_path)
        #new_conf[ind, :] = conf
        # img_file_path = os.path.join(LR_img_dctsmooth_base_path, img_files[ind])
        # LR_img = cv2.imread(img_file_path)
        # imgs.append(LR_img)
        # b, g, r = cv2.split(LR_img)
        # img = cv2.merge([r,g,b])
        #draw_point(joints, img, "/home/lgh/code/SMPLify/smplify_public/code/temp/temp_filter/img%04d" % ind)
        for i in range(14):
            array[ind, i*2] = LR_j2d[i, 0]
            array[ind, i*2+1] = LR_j2d[i, 1]

    for i in range(array.shape[1]):
        cv2.dct(array[:, i], fre)
        fre[dct_coe:] = 0
        cv2.idct(fre, ifre)
        for j in range(len(ifre)):
            if i % 2 == 0:
                new_joints[j, int(i / 2), 0] = ifre[j]
            else:
                new_joints[j, int(i / 2), 1] = ifre[j]
    joints = []
    for i in range(len(new_joints)):
        joints.append(new_joints[i])
    ###############save#################
    import matplotlib.pyplot as plt
    for i in range(length):
        x = new_joints[i, :, 0]
        y = new_joints[i, :, 1]
        plt.figure(1)
        plt.imshow(imgs[i])
        ax = plt.subplot(111)
        ax.scatter(x, y)
        plt.savefig(LR_img_dctsmooth_base_path + "/output/%04d.png" % i)
        plt.cla()
    return joints
        #plt.show()
    # write_j2d(new_joints, new_conf)
    #
    # for i in range(new_joints.shape[0]):
    #     img = cv2.imread("/home/lgh/code/SMPLify/smplify_public/code/temp/output/img_%04d.jpg" % i)
    #     b, g, r = cv2.split(img)
    #     img = cv2.merge([r, g, b])

def write_j2d(joints, conf):
    for i in range(conf.shape[0]):  # file number
        data = []
        _data = {"people": [{"pose_keypoints_2d": []}]}
        for j in range(conf.shape[1]):
            data.append(joints[i, j, 0])
            data.append(joints[i, j, 1])
            data.append(conf[i, j])
        _data["people"][0]["pose_keypoints_2d"] = data
        with open("/home/lgh/code/SMPLify/smplify_public/code/temp/temp_filter/img_%06d_render.json" % i,
                  'w') as json_file:
            json.dump(_data, json_file)


def load_dct_data():
    #HR_betas = load_HR_beta()
    LR_imgs = []
    j2ds = []
    poses = []
    betas = []
    trans = []
    cams = []
    params = []

    j2d_files = os.listdir(LR_j2d_base_path)
    j2d_files = sorted([filename for filename in j2d_files if filename.endswith(".npz")],
                       key=lambda d: int((d.split('_')[2]).split('.')[0]))
    img_files = os.listdir(LR_img_base_path)
    img_files = sorted([filename for filename in img_files if filename.endswith(".jpg")])

    pkl_files = os.listdir(LR_pkl_base_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                        key=lambda d: int(d.split('.')[0]))
    #start_ind = int(json_files[0].split('_')[1])
    for ind in range(0, BATCH_FRAME_NUM):
        j2d_file_path = os.path.join(LR_j2d_base_path, j2d_files[ind])
        LR_j2d, conf = load_deepcut(j2d_file_path)

        img_file_path = os.path.join(LR_img_base_path, img_files[ind])
        LR_img = cv2.imread(img_file_path)

        with open(os.path.join(LR_pkl_base_path, pkl_files[ind])) as f:
            param = pkl.load(f)
        params.append(param)

        LR_imgs.append(LR_img)

        np.array(LR_j2d)
        joints_i = LR_j2d[0:13, :]
        j2ds.append(joints_i)

        _pose = np.array(param['pose'])
        _pose = _pose[np.newaxis, :]
        poses.append(_pose)
        _beta = np.array(param['betas'])
        _beta = _beta[np.newaxis, :]
        betas.append(_beta)
        _tran = np.array(param['trans'])
        _tran = _tran[np.newaxis, :]
        trans.append(_tran)

        #####下面这个不确定参数的尺度，等之后debug看下
        cam = Perspective_Camera(param['f'][0] / 2 + 20, param['f'][1] / 2 + 20, img.shape[1] / 2,
                                    img.shape[0] / 2, param['t'], param['rt'])
        cams.append(cam)

    return imgs, j2ds, cams, poses, HR_betas, trans, params

def load_data_temporal(img_files):
    imgs = []; j2ds = []; poses = [];
    betas = []; trans = []
	
	
    for img_f in img_files:
        img_i, j2d_i_tmp, _, cam_i = load_data(img_f, NUM_VIEW)	    #返回一张图像三个视角的图像、2d pose、相机参数
        j2d_i = j2d_i_tmp[:, TEM_J2D_IDS]    #返回可供映射的13个2d pose
        imgs.append(img_i); j2ds.append(j2d_i);   #imgs，j2ds的每个元素代表一张图像的3个视角

        pose_path = img_f.replace('Image', 'Res_1')
        extension = os.path.splitext(pose_path)[1]
        pose_path = pose_path.replace(extension, '.pkl')    #代表每张图像的pkl的pose、beta、tran文件

        with open(pose_path, 'rb') as fin:
            res_1 = pkl.load(fin)
        poses.append( np.array(res_1['pose']) )
        betas.append( res_1['betas'] )
        trans.append( np.array(res_1['trans']) )

    mean_betas = np.array(betas)
    mean_betas = np.mean(mean_betas, axis=0)

    return imgs, j2ds, cam_i, poses, mean_betas, trans

def load_dct_base():
    mtx = sio.loadmat(DCT_MAT_PATH, squeeze_me=True, struct_as_record=False)
    mtx = mtx['D']
    mtx = mtx[:DCT_NUM]
    return np.array(mtx)

def load_initial_param():
    pose_prior = sio.loadmat(POSE_PRIOR_PATH, squeeze_me=True, struct_as_record=False)
    pose_mean = pose_prior['mean']
    pose_covariance = np.linalg.inv(pose_prior['covariance'])
    zero_shape = np.ones([13]) * 1e-8 # extra 3 for zero global rotation
    zero_trans = np.ones([3]) * 1e-8

    # pose_mean[45] = np.abs(pose_mean[45]) + 1.5
    # pose_mean[48] = np.abs(pose_mean[48]) + 1.5
    #
    # pose_mean[45] = 1e-8
    # pose_mean[46] = 1e-8
    # pose_mean[47] = -1.5
    # # pose_mean[48] = 1e-8
    # # pose_mean[49] = 1e-8
    # # pose_mean[50] = 1.4
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

## pose is N*69 array
def hr_means_covariance_lsw(pose):
    pose_mean = np.mean(pose, axis=0)
    ######covariance of error
    pose_error = pose - pose_mean
    covariance = np.ones([69,69]) * 1e-8
    for i in range(69):
        covariance[i, i] = pose_error[0, i]
    pose_covariance = np.linalg.inv(covariance)
    #covariance = np.cov(pose_error.T)
    #pose_covariance = np.linalg.inv(covariance)
    zero_shape = np.ones([13]) * 1e-8  # extra 3 for zero global rotation
    zero_trans = np.ones([3]) * 1e-8

    initial_param = np.concatenate([zero_shape, pose_mean, zero_trans], axis=0)
    dic = {"pose_mean": pose_mean, "pose_covariance" : pose_covariance,
                "initial_param" : initial_param, "pose_error" : pose_error}
    np.save("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/HR/output/init_param.npy", dic)
    return initial_param, pose_mean, pose_covariance

def load_hr_means_covariance_lsw():
    file = np.load("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/HR/output/init_param.npy").item()
    pose_mean = file["pose_mean"]
    pose_error = file["pose_error"]
    ######covariance of error
    covariance = np.ones([69, 69]) * 1e-8
    for i in range(69):
        covariance[i, i] = abs(pose_error[0, i])
    pose_covariance = np.linalg.inv(covariance)
    zero_shape = np.ones([13]) * 1e-8
    zero_trans = np.ones([3]) * 1e-8
    initial_param = np.concatenate([zero_shape, pose_mean, zero_trans], axis=0)
    #pose_covariance = file["pose_covariance"]
    #initial_param = file["initial_param"]
    return initial_param, pose_mean, pose_covariance
