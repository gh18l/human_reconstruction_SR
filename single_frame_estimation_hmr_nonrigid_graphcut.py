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
import pickle
import smpl_np
import scipy.io as sio
ind = 0

def get_nohandsfeet_weights(contours_smpl_correspondence, hands_feet_index):
    weights = np.ones_like(contours_smpl_correspondence)
    for i in range(len(contours_smpl_correspondence)):
        if contours_smpl_correspondence[i] in hands_feet_index:
            weights[i] = 0.0
    return weights


def get_n_min_index(values, n):
    mins = np.zeros(n, dtype=np.int)
    for i in range(n):
        min_index = np.argmin(values)
        mins[i] = min_index
        values[min_index] = 99999999.0
    return mins

##point -- 1*2 points -- 6890*2
def get_distance(point, points):
    distances = np.zeros([6890])
    for i in range(len(points)):
        distance = np.square(point[0, 0] - points[i, 0]) + np.square(point[0, 1] - points[i, 1])
        distances[i] = distance
    return distances

def find_maskcontours_smplcontours_correspondence(distances, min_smpl_index_old, _contours_smpl_index, i):
    old_index = smplindex_to_smplcontoursindex(min_smpl_index_old, _contours_smpl_index)
    min_smpl_index = np.argmin(distances)
    new_index = smplindex_to_smplcontoursindex(min_smpl_index, _contours_smpl_index)

    if old_index - i > len(_contours_smpl_index) / 2:   ##old_index is too big
        old_index = old_index - len(_contours_smpl_index)
    if new_index - i > len(_contours_smpl_index) / 2:  ##new_index is too big
        new_index = new_index - len(_contours_smpl_index)
    if i - old_index > len(_contours_smpl_index) / 2:  ##old_index is too small
        old_index = old_index + len(_contours_smpl_index)
    if i - new_index > len(_contours_smpl_index) / 2:  ##new_index is too small
        new_index = new_index + len(_contours_smpl_index)

    while (new_index - old_index) < 0 or (new_index - old_index) > 15:
        distances[min_smpl_index] = 99999999.0
        min_smpl_index = np.argmin(distances)
        new_index = smplindex_to_smplcontoursindex(min_smpl_index, _contours_smpl_index)
        if new_index - i > len(_contours_smpl_index) / 2:  ##new_index is too big
            new_index = new_index - len(_contours_smpl_index)
        if i - new_index > len(_contours_smpl_index) / 2:  ##new_index is too small
            new_index = new_index + len(_contours_smpl_index)
    return min_smpl_index


def get_distance_parsing(point, points, smpl_index):
    distances = np.ones([6890]) * 99999999.0
    for i in range(len(points)):
        if i in smpl_index:
            distance = np.square(point[0] - points[i, 0]) + np.square(point[1] - points[i, 1])
            distances[i] = distance
    return distances

def get_distance_parsing1(point, points, smpl_index):
    distances = np.ones([6890]) * 99999999.0
    for i in range(len(points)):
        if i in smpl_index:
            distance = np.square(point[0, 0] - points[i, 0]) + np.square(point[0, 1] - points[i, 1])
            distances[i] = distance
    return distances

def smplindex_to_smplcontoursindex(smpl_index, _contours_smpl_index):
    smpl_index_converted = []
    for i in range(len(_contours_smpl_index)):
        if smpl_index == _contours_smpl_index[i]:
            smpl_index_converted = i
            break
    return smpl_index_converted

def load_body_parsing():
    ### head+leftarm+rightarm   head+leftarm   head+rightarm   body+leftleg+rightleg
    ### body+leftleg   body+rightleg

    '''
    TODO
    make parsing similar to public datasets more
    '''
    dd = pickle.load(open(util.NORMAL_SMPL_PATH))
    weights = dd['weights']
    leg_index = np.array([1, 4, 7, 10, 2, 5, 8, 11])[:, np.newaxis]
    leftleg_index = np.array([1, 4, 7, 10])[:, np.newaxis]
    rightleg_index = np.array([2, 5, 8, 11])[:, np.newaxis]
    arm_index = np.array([17, 19, 21, 23, 16, 18, 20, 22, 14, 13])[:, np.newaxis]
    leftarm_index = np.array([16, 18, 20, 22, 13])[:, np.newaxis]
    rightarm_index = np.array([17, 19, 21, 23, 14])[:, np.newaxis]
    body_index = np.array([0, 3, 6, 9])[:, np.newaxis]
    head_index = np.array([12, 15])[:, np.newaxis]

    head_leftarm_rightarm_index = head_index
    head_leftarm_index = np.concatenate([head_index, leftarm_index], axis=0)
    head_rightarm_index = np.concatenate([head_index, rightarm_index], axis=0)
    body_leftleg_rightleg_index = np.concatenate([body_index, leftleg_index, rightleg_index],
                                                 axis=0)
    body_leftleg_index = np.concatenate([body_index, leftleg_index], axis=0)
    body_rightleg_index = np.concatenate([body_index, rightleg_index], axis=0)

    threshold = 0.3
    body_parsing_idx = []  ###body head
    _head_leftarm_rightarm_idx = np.zeros(6890)
    _head_leftarm_idx = np.zeros(6890)
    _head_rightarm_idx = np.zeros(6890)
    _body_leftleg_rightleg_idx = np.zeros(6890)
    _body_leftleg_idx = np.zeros(6890)
    _body_rightleg_idx = np.zeros(6890)
    placeholder_idx = np.zeros(6890)
    _test_idx = np.zeros(6890)

    for _, iii in enumerate(head_index):  ##head
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _head_leftarm_rightarm_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    head_leftarm_rightarm_idx = np.where(_head_leftarm_rightarm_idx == 1)
    body_parsing_idx.append(head_leftarm_rightarm_idx)

    for _, iii in enumerate(leftarm_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _head_leftarm_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    head_leftarm_idx = np.where(_head_leftarm_idx == 1)
    body_parsing_idx.append(head_leftarm_idx)

    for _, iii in enumerate(rightarm_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _head_rightarm_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    head_rightarm_idx = np.where(_head_rightarm_idx == 1)
    body_parsing_idx.append(head_rightarm_idx)

    for _, iii in enumerate(body_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _body_leftleg_rightleg_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    body_leftleg_rightleg_idx = np.where(_body_leftleg_rightleg_idx == 1)
    body_parsing_idx.append(body_leftleg_rightleg_idx)

    for _, iii in enumerate(leftleg_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _body_leftleg_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    body_leftleg_idx = np.where(_body_leftleg_idx == 1)
    body_parsing_idx.append(body_leftleg_idx)

    for _, iii in enumerate(rightleg_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _body_rightleg_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    body_rightleg_idx = np.where(_body_rightleg_idx == 1)
    body_parsing_idx.append(body_rightleg_idx)
    return body_parsing_idx

def load_body_parsing1():
    ### head+leftarm+rightarm   head+leftarm   head+rightarm   body+leftleg+rightleg
    ### body+leftleg   body+rightleg

    '''
    TODO
    make parsing similar to public datasets more
    '''
    dd = pickle.load(open(util.NORMAL_SMPL_PATH))
    weights = dd['weights']
    leg_index = np.array([1, 4, 7, 10, 2, 5, 8, 11])[:, np.newaxis]
    leftleg_index = np.array([1, 4, 7, 10])[:, np.newaxis]
    rightleg_index = np.array([2, 5, 8, 11])[:, np.newaxis]
    arm_index = np.array([17, 19, 21, 23, 16, 18, 20, 22, 14, 13])[:, np.newaxis]
    leftarm_index = np.array([16, 18, 20, 22, 13])[:, np.newaxis]
    rightarm_index = np.array([17, 19, 21, 23, 14])[:, np.newaxis]
    body_index = np.array([0, 3, 6, 9])[:, np.newaxis]
    head_index = np.array([12, 15])[:, np.newaxis]

    head_leftarm_rightarm_index = np.concatenate([head_index, leftarm_index, rightarm_index],
                                                 axis=0)
    head_leftarm_index = np.concatenate([head_index, leftarm_index], axis=0)
    head_rightarm_index = np.concatenate([head_index, rightarm_index], axis=0)
    body_leftleg_rightleg_index = np.concatenate([body_index, leftleg_index, rightleg_index],
                                                 axis=0)
    body_leftleg_index = np.concatenate([body_index, leftleg_index], axis=0)
    body_rightleg_index = np.concatenate([body_index, rightleg_index], axis=0)

    threshold = 0.3
    body_parsing_idx = []  ###body head
    _head_leftarm_rightarm_idx = np.zeros(6890)
    _head_leftarm_idx = np.zeros(6890)
    _head_rightarm_idx = np.zeros(6890)
    _body_leftleg_rightleg_idx = np.zeros(6890)
    _body_leftleg_idx = np.zeros(6890)
    _body_rightleg_idx = np.zeros(6890)

    _test_idx = np.zeros(6890)

    for _, iii in enumerate(head_leftarm_rightarm_index):  ##head
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold:
                _head_leftarm_rightarm_idx[ii] = 1
                _test_idx[ii] = 1
    head_leftarm_rightarm_idx = np.where(_head_leftarm_rightarm_idx == 1)
    body_parsing_idx.append(head_leftarm_rightarm_idx)

    for _, iii in enumerate(head_leftarm_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold:
                _head_leftarm_idx[ii] = 1
                _test_idx[ii] = 1
    head_leftarm_idx = np.where(_head_leftarm_idx == 1)
    body_parsing_idx.append(head_leftarm_idx)

    for _, iii in enumerate(head_rightarm_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold:
                _head_rightarm_idx[ii] = 1
                _test_idx[ii] = 1
    head_rightarm_idx = np.where(_head_rightarm_idx == 1)
    body_parsing_idx.append(head_rightarm_idx)

    for _, iii in enumerate(body_leftleg_rightleg_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold:
                _body_leftleg_rightleg_idx[ii] = 1
                _test_idx[ii] = 1
    body_leftleg_rightleg_idx = np.where(_body_leftleg_rightleg_idx == 1)
    body_parsing_idx.append(body_leftleg_rightleg_idx)

    for _, iii in enumerate(body_leftleg_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold:
                _body_leftleg_idx[ii] = 1
                _test_idx[ii] = 1
    body_leftleg_idx = np.where(_body_leftleg_idx == 1)
    body_parsing_idx.append(body_leftleg_idx)

    for _, iii in enumerate(body_rightleg_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold:
                _body_rightleg_idx[ii] = 1
                _test_idx[ii] = 1
    body_rightleg_idx = np.where(_body_rightleg_idx == 1)
    body_parsing_idx.append(body_rightleg_idx)
    return body_parsing_idx



def load_nonrigid_data():
    base_path = util.hmr_path + "output"
    pkl_files = os.listdir(base_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    poses = []
    betas = []
    trans = []
    cams = []
    for ind, pkl_file in enumerate(pkl_files):
        pkl_path = os.path.join(base_path, pkl_file)
        with open(pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        beta = param['betas']
        tran = param['trans']
        cam = param['cam_HR']
        poses.append(pose)
        betas.append(beta)
        trans.append(tran)
        cams.append(cam)
    return poses, betas, trans, cams

def load_parsing_mask():
    parsing_mask = cv2.imread(util.hmr_path + "optimization_data/label.png")
    return parsing_mask

def get_verts2d(cam, pose, beta, tran):
    param_shape = tf.constant(beta.reshape([1, -1]), dtype=tf.float32)
    param_rot = tf.constant(pose[0:3].reshape([1, -1]), dtype=tf.float32)
    param_pose = tf.constant(pose[3:72].reshape([1, -1]), dtype=tf.float32)
    param_trans = tf.constant(tran.reshape([1, -1]), dtype=tf.float32)
    initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
    cam_HR = Perspective_Camera(cam[0], cam[0], cam[1],
                                cam[2], cam[3], np.zeros(3))
    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)
    j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
    v = tf.reshape(v, [-1, 3])
    verts_est = cam_HR.project(tf.squeeze(v))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    verts_est1 = sess.run(verts_est)
    return verts_est1

def get_parsing_contours(parsing_mask):
    masks = np.zeros_like(parsing_mask)
    for i in range(parsing_mask.shape[0]):
        for j in range(parsing_mask.shape[1]):
            if parsing_mask[i, j, 0] == 0 and parsing_mask[i, j, 1] == 0 and parsing_mask[i, j, 2] == 0:
                continue
            masks[i, j, :] = 255
    mask = masks[:, :, 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)
    ## split contours into parsing
    mask_parsing_index = {'head': [], 'left_arm': [], 'right_arm': [],
                    'left_leg': [], 'right_leg': [], 'body': []}
    for i in range(len(contours[0])):
        color = parsing_mask[contours[0][i, :, 1], contours[0][i, :, 0]].squeeze()
        if color[0] == 0 and color[1] == 128 and color[2] == 0: ##green head
            mask_parsing_index['head'].append(i)
            mask_parsing_index['left_arm'].append(i)
            mask_parsing_index['right_arm'].append(i)
        if color[0] == 128 and color[1] == 0 and color[2] == 128:
            mask_parsing_index['right_arm'].append(i)
            mask_parsing_index['head'].append(i)
        if color[0] == 0 and color[1] == 128 and color[2] == 128:  ## yellow
            mask_parsing_index['left_arm'].append(i)
            mask_parsing_index['head'].append(i)
        if color[0] == 0 and color[1] == 0 and color[2] == 128: ##red
            mask_parsing_index['body'].append(i)
            mask_parsing_index['right_leg'].append(i)
            mask_parsing_index['left_leg'].append(i)
        if color[0] == 128 and color[1] == 128 and color[2] == 0:
            mask_parsing_index['right_leg'].append(i)
            mask_parsing_index['body'].append(i)
        if color[0] == 128 and color[1] == 0 and color[2] == 0: ##red
            mask_parsing_index['left_leg'].append(i)
            mask_parsing_index['body'].append(i)
    return mask_parsing_index, contours

def get_parsing_contours1(parsing_mask):
    masks = np.zeros_like(parsing_mask)
    for i in range(parsing_mask.shape[0]):
        for j in range(parsing_mask.shape[1]):
            if parsing_mask[i, j, 0] == 0 and parsing_mask[i, j, 1] == 0 and parsing_mask[i, j, 2] == 0:
                continue
            masks[i, j, :] = 255
    mask = masks[:, :, 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)
    return contours

def smpl_to_boundary(camera, pose, beta, tran, verts2d):
    smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    verts = smpl.get_verts(pose, beta, tran)
    bg = np.zeros([450, 600, 3], dtype=np.uint8)
    img_result_naked = camera.render_naked(verts, bg)
    mask = img_result_naked[:, :, 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ### find nearest 2d smpl point of each contour point to get boundary
    '''
    TODO
    each contour ---> smplverts, to make more smpl verts being boundary
    '''
    contours_smpl_index = []
    for ind in range(len(contours)):
        if len(contours[ind]) < 100:
            continue
        for i in range(len(contours[ind])):
            contour = contours[ind][i, :, :]
            ### if it need faster, the search range can be limited in small district
            distances = get_distance(contour, verts2d)
            min_smpl_indexs = get_n_min_index(distances, 3)
            #min_smpl_index = np.argmin(distances)
            for n in range(len(min_smpl_indexs)):
                contours_smpl_index.append(min_smpl_indexs[n])
    contours_smpl_index = np.unique(np.array(contours_smpl_index))
    # ## view smpl contours result
    # for i in range(len(contours_smpl_index)):
    #     smpl_index = contours_smpl_index[i]
    #     x = np.rint(verts2d[smpl_index, 0]).astype("int")
    #     y = np.rint(verts2d[smpl_index, 1]).astype("int")
    #     mask[y, x] = 255
    #
    # cv2.imshow("1", mask)
    # cv2.waitKey()
    return contours_smpl_index

def smpl_to_boundary1(camera, pose, beta, tran, verts2d):
    smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    verts = smpl.get_verts(pose, beta, tran)
    bg = np.zeros([1920, 1080, 3], dtype=np.uint8)
    img_result_naked = camera.render_naked(verts, bg)
    mask = img_result_naked[:, :, 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntsSorted = sorted(contours[0], key=lambda x: cv2.contourArea(x))

    ## view
    # for i in range(len(contours[0])):
    #     x = np.rint(contours[0][i, :, 0]).astype("int")
    #     y = np.rint(contours[0][i, :, 1]).astype("int")
    #     mask[y, x] = 255
    #     cv2.imshow("1", mask)
    #     cv2.waitKey()

    ### find nearest 2d smpl point of each contour point to get boundary
    '''
    TODO
    each contour ---> smplverts, to make more smpl verts being boundary
    '''
    contours_smpl_index = []
    for ind in range(len(contours)):
        # if len(contours[ind]) < 100:
        #     continue
        for i in range(len(contours[ind])):
            contour = contours[ind][i, :, :]
            ### if it need faster, the search range can be limited in small district
            distances = get_distance(contour, verts2d)
            min_smpl_indexs = get_n_min_index(distances, 1)
            #min_smpl_index = np.argmin(distances)
            for n in range(len(min_smpl_indexs)):
                contours_smpl_index.append(min_smpl_indexs[n])
    contours_smpl_index1 = list(set(contours_smpl_index))
    contours_smpl_index1.sort(key=contours_smpl_index.index)
    contours_smpl_index = contours_smpl_index1
    #contours_smpl_index = np.unique(np.array(contours_smpl_index))
    ## view smpl contours result
    # for i in range(len(contours_smpl_index)):
    #     if i < 700:
    #         continue
    #     smpl_index = contours_smpl_index[i]
    #     x = np.rint(verts2d[smpl_index, 0]).astype("int")
    #     y = np.rint(verts2d[smpl_index, 1]).astype("int")
    #     mask[y, x] = 255
    #     cv2.imshow("1", mask)
    #     cv2.waitKey()

    return contours_smpl_index

def get_parsing_smpl_contours(contours_smpl_index, body_parsing_idx):
    smpl_parsing_index = {'head': [], 'right_arm': [], 'left_arm': [],
                          'body': [], 'right_leg': [], 'left_leg': []}
    for i in range(len(contours_smpl_index)):
        index = contours_smpl_index[i]
        if index in body_parsing_idx[0][0]:  ###head_leftarm_rightarm
            smpl_parsing_index["head"].append(index)
        if index in body_parsing_idx[1][0]:  ###head_leftarm
            smpl_parsing_index["left_arm"].append(index)
        if index in body_parsing_idx[2][0]:  ###head_rightarm
            smpl_parsing_index["right_arm"].append(index)
        if index in body_parsing_idx[3][0]:  ###body_leftleg_rightleg
            smpl_parsing_index["body"].append(index)
        if index in body_parsing_idx[4][0]:  ###body_leftleg
            smpl_parsing_index["left_leg"].append(index)
        if index in body_parsing_idx[5][0]:  ###body_rightleg
            smpl_parsing_index["right_leg"].append(index)
    return smpl_parsing_index

# def get_maskcontours_smpl_index(mask_parsing_index, mask_contours, contours_smpl_index, verts2d,
#                                 _contours_smpl_index, hands_feet_index):
#     '''
#     count number of array in mask_contours
#     :param mask_parsing_index:
#     :param mask_contours:
#     :param contours_smpl_index:
#     :param verts2d:
#     :return: the corresponding smpl index of each mask contour
#     '''
#     length = 0
#     for name in mask_parsing_index:
#         length = length + len(mask_parsing_index[name])
#     min_smpl_index_old = []
#     maskcontours_smpl_index = np.zeros(length)
#     mask_weights = np.ones(length)
#     for i in range(len(mask_contours[0])):
#         for name in mask_parsing_index:
#             if i in mask_parsing_index[name]:
#                 smpl_indexs = contours_smpl_index[name]
#                 distances = get_distance_parsing1(mask_contours[0, i, :, :], verts2d, smpl_indexs)
#                 if i == 0:
#                     min_smpl_index = np.argmin(distances)
#                     min_smpl_index_old = min_smpl_index
#                 if i != 0:
#                     min_smpl_index = find_maskcontours_smplcontours_correspondence(distances, min_smpl_index_old, _contours_smpl_index, i)
#                     min_smpl_index_old = min_smpl_index
#                 maskcontours_smpl_index[i] = min_smpl_index
#                 ## view smpl contours result
#                 if i < 750:
#                     continue
#                 bg = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/output/hmr_optimization_0004.png")
#                 smpl_index = maskcontours_smpl_index[i].astype("int")
#                 x = np.rint(verts2d[smpl_index, 0]).astype("int")
#                 y = np.rint(verts2d[smpl_index, 1]).astype("int")
#                 bg[y, x, 0] = 255
#
#                 x = mask_contours[0, i, :, 0].astype("int")
#                 y = mask_contours[0, i, :, 1].astype("int")
#                 bg[y, x, 2] = 255
#                 new_index = smplindex_to_smplcontoursindex(maskcontours_smpl_index[i], _contours_smpl_index)
#                 print(new_index)
#                 cv2.imshow("1", bg)
#                 cv2.waitKey()
#
#     return maskcontours_smpl_index, mask_weights


def get_maskcontours_smpl_index(mask_parsing_index, mask_contours, contours_smpl_index, verts2d,
                                _contours_smpl_index, hands_feet_index):
    '''
    count number of array in mask_contours
    :param mask_parsing_index:
    :param mask_contours:
    :param contours_smpl_index:
    :param verts2d:
    :return: the corresponding smpl index of each mask contour
    '''
    length = 0
    for name in mask_parsing_index:
        length = length + len(mask_parsing_index[name])

    maskcontours_smpl_index = np.zeros(length)
    mask_weights = np.ones(length)
    for name in mask_parsing_index:
        smpl_indexs = contours_smpl_index[name]
        for i in range(len(mask_parsing_index[name])):
            mask_index = mask_parsing_index[name][i]
            distances = get_distance_parsing1(mask_contours[0][mask_index, :, :], verts2d, smpl_indexs)
            min_smpl_index = np.argmin(distances)
            smpl_index_converted = smplindex_to_smplcontoursindex(min_smpl_index, _contours_smpl_index)
            maskcontours_smpl_index[mask_index] = min_smpl_index
            if min_smpl_index in hands_feet_index:
                mask_weights[mask_index] = 0.0
    return maskcontours_smpl_index, mask_weights

def get_smplcontours_mask_index(mask_parsing_index, mask_contours, contours_smpl_index, verts2d, _contours_smpl_index,
                                hands_feet_index):
    length = 0
    for name in contours_smpl_index:
        length = length + len(contours_smpl_index[name])

    smplcontours_mask_index = np.zeros(length, dtype=np.int64)
    smpl_weights = np.zeros(length)
    for name in contours_smpl_index:
        mask_indexs = mask_parsing_index[name]
        for i in range(len(contours_smpl_index[name])):
            smpl_index = contours_smpl_index[name][i]
            smpl_index_converted = smplindex_to_smplcontoursindex(smpl_index, _contours_smpl_index)
            distances = get_distance_parsing(verts2d[smpl_index, :], mask_contours.squeeze(), mask_indexs)
            min_mask_index = np.argmin(distances)
            smplcontours_mask_index[smpl_index_converted] = min_mask_index
            if smpl_index in hands_feet_index:
                smpl_weights[smpl_index_converted] = 1.0
    return smplcontours_mask_index, smpl_weights

def nonrigid_estimation():
    hmr_dict, data_dict = util.load_hmr_data(util.hmr_path)
    HR_imgs = data_dict["imgs"]
    HR_masks = data_dict["masks"]
    hands_feet_index = opt_pre.get_hands_feet_index()
    poses, betas, trans, cams = load_nonrigid_data()
    body_parsing_idx = load_body_parsing()
    body_parsing_idx1 = load_body_parsing1()
    parsing_mask = load_parsing_mask()

    camera = render.camera(cams[ind][0], cams[ind][1], cams[ind][2], cams[ind][3])
    ## get parsing img contours index
    ## 1 is normal, no 1 is hands and feet
    verts2d = get_verts2d(cams[ind], poses[ind], betas[ind], trans[ind])
    # mask_parsing_index, contours = get_parsing_contours(parsing_mask)
    contours1 = get_parsing_contours1(parsing_mask)
    # contours_smpl_index = smpl_to_boundary(camera, poses[ind], betas[ind], trans[ind], verts2d)
    contours_smpl_index1 = smpl_to_boundary1(camera, poses[ind], betas[ind], trans[ind], verts2d)
    contours_smpl_index1 = np.array(contours_smpl_index1)
    #write_file(contours1, verts2d, contours_smpl_index1)
    label_result = sio.loadmat(util.hmr_path + "optimization_data/label_result.mat")
    label_result = label_result["label_result"].astype("int").squeeze()
    ##### generate smpl shape template
    param_shape = tf.Variable(betas[ind].reshape([1, -1]), dtype=tf.float32)
    param_rot = tf.constant(poses[ind][0:3].reshape([1, -1]), dtype=tf.float32)
    param_pose = tf.constant(poses[ind][3:72].reshape([1, -1]), dtype=tf.float32)
    param_trans = tf.constant(trans[ind].reshape([1, -1]), dtype=tf.float32)

    ####tensorflow array initial_param_tf
    smpl_model = SMPL(util.SMPL_PATH, util.NORMAL_SMPL_PATH)
    initial_param_tf = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
    j3ds, v, jointsplus = smpl_model.get_3d_joints(initial_param_tf, util.SMPL_JOINT_IDS)
    v_shaped_tf = tf.reshape(v, [-1, 3])

    smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    v = smpl.get_verts(poses[ind], betas[ind], trans[ind])
    v_tf = tf.Variable(v, dtype=tf.float32)

    ### convert v_tf to laplace coordination
    faces = camera.renderer.faces.astype(np.int64)
    L = opt_pre.get_laplace_operator(v_shaped_tf, faces)
    delta = tf.matmul(L, v_shaped_tf)
    weights_laplace = opt_pre.get_laplace_weights()
    weights_laplace = 4.0 * weights_laplace.reshape(-1, 1)


    cam_nonrigid = Perspective_Camera(cams[ind][0], cams[ind][0], cams[ind][1],
                                      cams[ind][2], cams[ind][3], np.zeros(3))
    verts_est = cam_nonrigid.project(tf.squeeze(v_tf))
    objs_nonrigid = {}

    contours_smpl_correspondence = contours_smpl_index1[label_result-1]
    weights = get_nohandsfeet_weights(contours_smpl_correspondence, hands_feet_index)

    ### view
    # for i in range(len(contours_smpl_correspondence)):
    #     if weights[i] == 0.0:
    #         index = contours_smpl_correspondence[i]
    #         x = verts2d[index, 0].astype("int")
    #         y = verts2d[index, 1].astype("int")
    #         HR_imgs[ind][y, x, 0] = 255
    #     else:
    #         index = contours_smpl_correspondence[i]
    #         x = verts2d[index, 0].astype("int")
    #         y = verts2d[index, 1].astype("int")
    #         HR_imgs[ind][y, x, 2] = 255
    # cv2.imshow("1", HR_imgs[ind])
    # cv2.waitKey()

    contours_smpl_correspondence = contours_smpl_correspondence.reshape([-1, 1]).astype(np.int64)

    verts_est1 = tf.gather_nd(verts_est, contours_smpl_correspondence)
    objs_nonrigid['verts_loss'] = 0.08 * tf.reduce_sum(weights * tf.reduce_sum(tf.square(verts_est1 - contours1.squeeze()), 1))

    #### verts_loss1
    #maskcontours_smpl_index = maskcontours_smpl_index.reshape([-1, 1]).astype(np.int64)
    #verts_est_contours = tf.gather_nd(verts_est, maskcontours_smpl_index)
    #objs_nonrigid['verts_loss1'] = 0.08 * tf.reduce_sum(tf.reduce_sum(tf.square(verts_est_contours - contours.squeeze()), 1))


    #### norm choose   weights_laplace
    objs_nonrigid['laplace'] = 0.1 * tf.reduce_sum(weights_laplace * tf.reduce_sum(tf.square(tf.matmul(L, v_tf) - delta), 1))
    objs_nonrigid['smooth_loss'] = 1.0 * tf.reduce_sum(tf.square(v_tf - v_shaped_tf))

    loss = tf.reduce_mean(objs_nonrigid.values())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # L-BFGS-B
        optimizer = scipy_pt(loss=loss, var_list=[v_tf, param_shape],
                             options={'eps': 1e-12, 'ftol': 1e-12, 'maxiter': 1000, 'disp': False}, method='L-BFGS-B')
        optimizer.minimize(sess)
        v_nonrigid_final = sess.run(v_tf)
        verts2d = sess.run(verts_est)
        _objs_nonrigid = sess.run(objs_nonrigid)
        pose_final, betas_final, trans_final = sess.run(
            [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])
        for name in _objs_nonrigid:
            print("the %s loss is %f" % (name, _objs_nonrigid[name]))

    ### view data
    ## dont render fingers
    with open("/home/lgh/code/SMPLify_TF/smpl/models/bodyparts.pkl", 'rb') as f:
        v_ids = pickle.load(f)
    fingers = np.concatenate((v_ids['fingers_r'], v_ids['fingers_l']))
    faces = camera.renderer.faces
    camera.renderer.faces = np.array(filter(lambda face: np.intersect1d(face, fingers).size == 0, faces))

    _, vt = camera.generate_uv(v_nonrigid_final, HR_imgs[ind])
    if not os.path.exists(util.hmr_path + "output_nonrigid"):
        os.makedirs(util.hmr_path + "output_nonrigid")
    img_result_texture = camera.render_texture(v_nonrigid_final, HR_imgs[ind], vt)
    cv2.imwrite(util.hmr_path + "output_nonrigid/hmr_optimization_texture_nonrigid%04d.png" % ind, img_result_texture)
    img_result_naked = camera.render_naked(v_nonrigid_final, HR_imgs[ind])
    cv2.imwrite(util.hmr_path + "output_nonrigid/hmr_optimization_nonrigid_%04d.png" % ind, img_result_naked)
    img_result_naked_rotation = camera.render_naked_rotation(v_nonrigid_final, 90, HR_imgs[ind])
    cv2.imwrite(util.hmr_path + "output_nonrigid/hmr_optimization_rotation_nonrigid_%04d.png" % ind, img_result_naked_rotation)
    camera.write_obj(util.hmr_path + "output_nonrigid/hmr_optimization_rotation_nonrigid_%04d.obj" % ind, v_nonrigid_final, vt)
    camera.write_texture_data(util.texture_path, HR_imgs[ind], vt)
    template = smpl.get_nonrigid_smpl_template(v_nonrigid_final, pose_final, betas_final, trans_final)
    render.save_nonrigid_template(util.texture_path, template)

    for z in range(len(verts2d)):
        if int(verts2d[z][0]) > HR_masks[ind].shape[1] - 1:
            print(int(verts2d[z][0]))
            verts2d[z][0] = HR_masks[ind].shape[1] - 1
        if int(verts2d[z][1]) > HR_masks[ind].shape[0] - 1:
            print(int(verts2d[z][1]))
            verts2d[z][1] = HR_masks[ind].shape[0] - 1
        (HR_masks[ind])[int(verts2d[z][1]), int(verts2d[z][0])] = 127
    if not os.path.exists(util.hmr_path + "output_mask"):
        os.makedirs(util.hmr_path + "output_mask")
    cv2.imwrite(util.hmr_path + "output_mask/%04d_nonrigid.png" % ind, HR_masks[ind])
nonrigid_estimation()