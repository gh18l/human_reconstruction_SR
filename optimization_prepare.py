import tensorflow as tf
import numpy as np
import pickle as pkl
from scipy import sparse as sp
import cv2
import pickle
import util
import copy
def get_tf_mask(verts_est, HR_mask):
    verts_est = tf.cast(verts_est, dtype=tf.int64)
    verts_est = tf.concat([tf.expand_dims(verts_est[:, 1], 1),
                           tf.expand_dims(verts_est[:, 0], 1)], 1)

    verts_est_shape = verts_est.get_shape().as_list()
    temp_np = np.ones([verts_est_shape[0]]) * 255
    temp_np = tf.convert_to_tensor(temp_np, dtype=tf.float32)
    delta_shape = tf.convert_to_tensor([HR_mask.shape[0], HR_mask.shape[1]],
                                       dtype=tf.int64)
    scatter = tf.scatter_nd(verts_est, temp_np, delta_shape)
    compare = np.zeros([HR_mask.shape[0], HR_mask.shape[1]])
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

    return verts2dsilhouette

def get_laplace_weights():
    weights = np.ones(6890)
    with open('./smpl/models/bodyparts.pkl', 'rb') as f:
        v_ids = pkl.load(f)

    weights[v_ids['face']] = 12.
    weights[v_ids['hand_l']] = 5.
    weights[v_ids['hand_r']] = 5.
    weights[v_ids['fingers_l']] = 8.  ##8.
    weights[v_ids['fingers_r']] = 8.  ##8.
    weights[v_ids['foot_l']] = 5.
    weights[v_ids['foot_r']] = 5.
    weights[v_ids['toes_l']] = 8.  ##8.
    weights[v_ids['toes_r']] = 8.  ##8.
    weights[v_ids['ear_l']] = 10.
    weights[v_ids['ear_r']] = 10.
    return weights

def get_model_weights():
    weights = np.ones(6890)
    with open('./smpl/models/bodyparts.pkl', 'rb') as f:
        v_ids = pkl.load(f)

    weights[v_ids['face']] = 7.
    weights[v_ids['hand_l']] = 12.
    weights[v_ids['hand_r']] = 12.
    weights[v_ids['fingers_l']] = 15.
    weights[v_ids['fingers_r']] = 15.
    weights[v_ids['foot_l']] = 12.
    weights[v_ids['foot_r']] = 12.
    weights[v_ids['toes_l']] = 15.
    weights[v_ids['toes_r']] = 15.
    weights[v_ids['ear_l']] = 10.
    weights[v_ids['ear_r']] = 10.

    return weights

### v is a tensor
def get_laplace_operator(v, f):
    n = list(v.get_shape())[0]
    v_a = f[:, 0].reshape([-1, 1])
    v_b = f[:, 1].reshape([-1, 1])
    v_c = f[:, 2].reshape([-1, 1])


    ab = tf.gather_nd(v, v_a) - tf.gather_nd(v, v_b)
    bc = tf.gather_nd(v, v_b) - tf.gather_nd(v, v_c)
    ca = tf.gather_nd(v, v_c) - tf.gather_nd(v, v_a)

    cot_a = -1 * tf.reduce_sum(ab * ca, 1) / tf.sqrt(tf.reduce_sum(tf.cross(ab, ca) ** 2, -1))
    cot_b = -1 * tf.reduce_sum(bc * ab, 1) / tf.sqrt(tf.reduce_sum(tf.cross(bc, ab) ** 2, -1))
    cot_c = -1 * tf.reduce_sum(ca * bc, 1) / tf.sqrt(tf.reduce_sum(tf.cross(ca, bc) ** 2, -1))

    I = tf.concat([v_a, v_c, v_a, v_b, v_b, v_c], 0)
    J = tf.concat([v_c, v_a, v_b, v_a, v_c, v_b], 0)
    W = 0.5 * tf.concat([cot_b, cot_b, cot_c, cot_c, cot_a, cot_a], 0)
    indices = tf.concat([I, J], 1)
    sparseL = tf.SparseTensor(indices=indices, values=W, dense_shape=[n, n])
    L = tf.sparse_tensor_to_dense(sparseL, validate_indices=False)
    L = L - tf.matrix_diag(tf.squeeze(tf.matmul(L, tf.ones([n, 1]))))
    return L

def load_body_parsing():
    dd = pickle.load(open(util.NORMAL_SMPL_PATH))
    weights = dd['weights']
    #leg_index = [1, 4, 7, 10, 2, 5, 8, 11]
    leg_index = []
    arm_index = [17, 19, 21, 23, 16, 18, 20, 22, 14, 13]
    body_index = [3, 6, 9]
    #body_index = []
    head_index = [12, 15]
    threshold = 0.6
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
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _body_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    body_idx = np.where(_body_idx == 1)
    body_parsing_idx.append(body_idx)

    for _, iii in enumerate(head_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _head_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    head_idx = np.where(_head_idx == 1)
    body_parsing_idx.append(head_idx)

    for _, iii in enumerate(leg_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _leg_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    leg_idx = np.where(_leg_idx == 1)
    body_parsing_idx.append(leg_idx)

    for _, iii in enumerate(arm_index):
        length = len(weights[:, iii])
        for ii in range(length):
            if weights[ii, iii] > threshold and placeholder_idx[ii] == 0:
                _arm_idx[ii] = 1
                placeholder_idx[ii] = 1
                _test_idx[ii] = 1
    arm_idx = np.where(_arm_idx == 1)
    body_parsing_idx.append(arm_idx)
    return body_parsing_idx


def get_dense_correspondence(verts, img1, img2):
    flow = np.array([])
    retval = cv2.DISOpticalFlow_create(2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ### the flow from img1 to img2
    flow = retval.calc(img1, img2, flow)
    verts_new = copy.deepcopy(verts)
    ## x y
    for i in range(len(verts)):
        position_x = verts[i][0]
        position_y = verts[i][1]
        y = np.round(position_y).astype(np.int)
        x = np.round(position_x).astype(np.int)
        verts_new[i][0] = position_x + flow[y, x, 0]  ##x
        verts_new[i][1] = position_y + flow[y, x, 1]  ##y
    return verts_new

def get_hands_feet_index():
    with open('./smpl/models/bodyparts.pkl', 'rb') as fp:
        v_ids = pkl.load(fp)
    hands_feet = np.concatenate((v_ids['fingers_r'], v_ids['fingers_l']))
    #hands_feet = []
    return hands_feet

