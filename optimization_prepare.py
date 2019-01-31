import tensorflow as tf
import numpy as np
import pickle as pkl
from scipy import sparse as sp
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
    weights[v_ids['fingers_l']] = 8.
    weights[v_ids['fingers_r']] = 8.
    weights[v_ids['foot_l']] = 5.
    weights[v_ids['foot_r']] = 5.
    weights[v_ids['toes_l']] = 8.
    weights[v_ids['toes_r']] = 8.
    weights[v_ids['ear_l']] = 10.
    weights[v_ids['ear_r']] = 10.
    return weights

### v is a tensor
def get_laplace_operator(v, f):
    n = list(v.get_shape())[0]
    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    ab = v[v_a] - v[v_b]
    bc = v[v_b] - v[v_c]
    ca = v[v_c] - v[v_a]

    cot_a = -1 * tf.reduce_sum(ab * ca, 1) / tf.sqrt(tf.reduce_sum(tf.cross(ab, ca) ** 2, -1))
    cot_b = -1 * tf.reduce_sum(bc * ab, 1) / tf.sqrt(tf.reduce_sum(tf.cross(bc, ab) ** 2, -1))
    cot_c = -1 * tf.reduce_sum(ca * bc, 1) / tf.sqrt(tf.reduce_sum(tf.cross(ca, bc) ** 2, -1))

    I = tf.concat([v_a, v_c, v_a, v_b, v_b, v_c], 0)
    J = tf.concat([v_c, v_a, v_b, v_a, v_c, v_b], 0)
    W = 0.5 * tf.concat([cot_b, cot_b, cot_c, cot_c, cot_a, cot_a], 0)

    L = tf.SparseTensor(indices=[I, J], values=W, dense_shape=[n, n])

    L = L - tf.matrix_diag(L * tf.ones(n))
    return L
