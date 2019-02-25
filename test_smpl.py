import pickle as pkl
import smpl_np
from opendr_render import render
import cv2
import numpy as np
import util
path = util.hmr_path + "output_nonrigid/hmr_optimization_pose_%04d.pkl" % 41
with open(path) as f:
    param = pkl.load(f)
verts = np.load(util.hmr_path + "output_nonrigid/hmr_optimization_pose_%04d.npy" % 41)
beta = param['betas'].squeeze()
tran = param['trans'].squeeze()
pose = param['pose'].squeeze()
cam = param['cam_HR']

img = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/optimization_data/0041.jpg")

smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
verts = verts - tran.reshape([1, 3])
verts_template = smpl.get_nonrigid_smpl_template(verts, pose, beta, tran)
smpl.set_template(verts_template)
template = smpl.get_verts(pose, beta, tran)
camera = render.camera(cam[0], cam[1], cam[2], cam[3])

bg = np.zeros_like(img)

img_result_naked = camera.render_naked(template, bg)
# = camera.render_naked(verts_template)
camera.write_obj(util.hmr_path + "output_nonrigid/hmr_optimization_pose_%04d.obj" % 41, verts_template)
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


