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
body_index = [0, 3, 6, 9]
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
        if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
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

##