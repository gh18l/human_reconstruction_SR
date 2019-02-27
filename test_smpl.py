import pickle as pkl
import smpl_np
from opendr_render import render
import cv2
import numpy as np
import util
import pickle
path = util.hmr_path + "output_nonrigid_old/hmr_optimization_pose_%04d.pkl" % 41
with open(path) as f:
    param = pkl.load(f)
verts = np.load(util.hmr_path + "output_nonrigid_old/hmr_optimization_pose_%04d.npy" % 41)
beta = param['betas'].squeeze()
tran = param['trans'].squeeze()
pose = param['pose'].squeeze()
cam = param['cam_HR']

# img = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/optimization_data/0041.jpg")
# dd = pickle.load(open(util.NORMAL_SMPL_PATH))
# weights = dd['weights']
# vert_sym_idxs = dd['vert_sym_idxs']
# v_template = dd['v_template']
# leg_index = [1, 4, 7, 10, 2, 5, 8, 11]
# arm_index = [17, 19, 21, 23, 16, 18, 20, 22, 14, 13]
# body_index = [21, 23]
# head_index = [12, 15]
# body_parsing_idx = []  ###body head
# _leg_idx = np.zeros(6890)
# _arm_idx = np.zeros(6890)
# _body_idx = np.zeros(6890)
# _head_idx = np.zeros(6890)
# placeholder_idx = np.zeros(6890)
# _test_idx = np.zeros(6890)
#
# for _, iii in enumerate(body_index):
#     length = len(weights[:, iii])
#     for ii in range(length):
#         if weights[ii, iii] > 0.4 and placeholder_idx[ii] == 0:
#             _body_idx[ii] = 1
#             placeholder_idx[ii] = 1
#             _test_idx[ii] = 1
# body_idx = np.where(_body_idx == 1)
# body_parsing_idx.append(body_idx)
#
# for _, iii in enumerate(head_index):
#     length = len(weights[:, iii])
#     for ii in range(length):
#         if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
#             _head_idx[ii] = 1
#             placeholder_idx[ii] = 1
#             _test_idx[ii] = 1
# head_idx = np.where(_head_idx == 1)
# body_parsing_idx.append(head_idx)
#
# for _, iii in enumerate(leg_index):
#     length = len(weights[:, iii])
#     for ii in range(length):
#         if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
#             _leg_idx[ii] = 1
#             placeholder_idx[ii] = 1
#             _test_idx[ii] = 1
# leg_idx = np.where(_leg_idx == 1)
# body_parsing_idx.append(leg_idx)
#
# for _, iii in enumerate(arm_index):
#     length = len(weights[:, iii])
#     for ii in range(length):
#         if weights[ii, iii] > 0.4 and placeholder_idx[ii] == 0:
#             _arm_idx[ii] = 1
#             placeholder_idx[ii] = 1
#             _test_idx[ii] = 1
# arm_idx = np.where(_arm_idx == 1)
# body_parsing_idx.append(arm_idx)
#
# with open("./smpl/models/bodyparts.pkl",'rb') as f:
#     v_ids = pkl.load(f)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1)
# #ax = plt.subplot(111)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(v_template[body_parsing_idx[0], 0], v_template[body_parsing_idx[0], 1], v_template[body_parsing_idx[0], 2], c='b')
# #ax.scatter(v_template[body_parsing_idx[2], 0], v_template[body_parsing_idx[2], 1], v_template[body_parsing_idx[2], 2], c='b')
# #ax.scatter(v_template[body_parsing_idx[1], 0], v_template[body_parsing_idx[1], 1], v_template[body_parsing_idx[1], 2], c='b')
# #ax.scatter(v_template[v_ids['hand_l'], 0], v_template[v_ids['hand_l'], 1], v_template[v_ids['hand_l'], 2], c='g')
# #ax.scatter(v_template[v_ids['hand_r'], 0], v_template[v_ids['hand_r'], 1], v_template[v_ids['hand_r'], 2], c='y')
# ax.scatter(v_template[v_ids['hand_l'], 0], v_template[v_ids['hand_l'], 1], v_template[v_ids['hand_l'], 2], c='c')
# ax.scatter(v_template[:, 0], v_template[:, 1], v_template[:, 2], c='r', s=1)
# plt.show()
smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
template = smpl.remove_template_handfoot()
# min = 99999999
# indexxx = 0
# z_mm = []
# for i in range(len(a[0])):
#     index = a[0][i]
#     #if template[index, 0] >0.7 and template[index, 0] <0.72:
#     z_mm.append(template[index, 2])
# z_mm = np.array(z_mm).squeeze()
# z_min = np.min(z_mm)
# z_max = np.max(z_mm)
# for i in range(len(a[0])):
#     index = a[0][i]
#     z = template[index, 2]
#     if z > z_max:
#         template[index, 2] = z_max
#     if z < z_min:
#         template[index, 2] = z_min
# for i in range(len(a[0])):
#     index = a[0][i]
#     x = template[index, 0]
#     if x < min:
#         min = x
#         indexxx = index
# for i in range(len(a[0])):
#     index = a[0][i]
#     template[index, 0] = min

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
# ax = plt.subplot(111)
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(v_template[index, 0], v_template[index, 1], v_template[index, 2], c='b')
#ax.scatter(v_template[v_ids['fingers_l'], 0], v_template[v_ids['fingers_l'], 1], v_template[v_ids['fingers_l'], 2], c='c')
ax.scatter(template[:, 0], template[:, 1], template[:, 2], c='r', s=1)
#print(v_template[index, 0])
plt.show()

camera = render.camera(cam[0], cam[1], cam[2], cam[3])

bg = np.zeros_like(img)

img_result_naked = camera.render_naked(template, img)
# = camera.render_naked(verts_template)
#camera.write_obj(util.hmr_path + "output_nonrigid/hmr_optimization_pose_%04d.obj" % 41, verts_template)
cv2.imwrite("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/output_nonrigid/test.png", img_result_naked)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
#ax = plt.subplot(111)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(verts_template[:, 0], verts_template[:, 1], verts_template[:, 2], c='b')
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='r')
# ax.scatter(j2ds_est1[14, 0], j2ds_est1[14, 1], c='r')
# #ax.scatter(HR_j2ds_foot[ind][0, 0], HR_j2ds_foot[ind][0, 1], c='b')
# #hmr_joint3d = hmr_joint3ds[ind,:,:]
# #ax.scatter(j3ds1[14, 0], j3ds1[14, 1], j3ds1[14, 2], c='r',s=40)
# #ax.scatter(j3ds1[13, 0], j3ds1[13, 1], j3ds1[13, 2], c='r',s=40)
# plt.imshow(HR_imgs[ind])
plt.show()
# #plt.imshow(HR_imgs[ind])
# #ax.scatter(verts_est1[:, 0], verts_est1[:, 1], c='r')
# #plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png")


