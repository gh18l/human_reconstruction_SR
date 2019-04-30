"""Render a model."""
import cPickle as pickle
import logging as _logging
import os as _os
#from render_model import render_model
from smpl.smpl_webuser.lbs import global_rigid_transformation
#import os.path as _path
# pylint: disable=invalid-name
#import sys as _sys
import sys
from copy import copy
#import chumpy as ch
import cv2
import numpy as np
import util
#from up_tools.mesh import Mesh
try:
    from smpl.serialization import load_model as _load_model
except:
    from smpl.smpl_webuser.serialization import load_model as _load_model

_LOGGER = _logging.getLogger(__name__)
#_TEMPLATE_MESH = Mesh(filename=_os.path.join(_os.path.dirname(__file__),
                                              #'models', '3D', 'template.ply'))
_COLORS = {
    'pink': [.6, .6, .8],
    'cyan': [.7, .75, .5],
    'yellow': [.5, .7, .75],
    'grey': [.7, .7, .7],
}

def generate_uv(model, image, cam):
    beta = np.squeeze(cam['betas'])
    model.betas[:len(beta)] = beta
    pose = np.squeeze(cam['pose'])
    model.pose[:] = pose
    tran = np.squeeze(cam['trans'])
    model.trans[:] = tran
    w, h = (image.shape[1], image.shape[0])
    f = np.array([w, w]) / 2. if cam['f'] is None else cam['f']
    c = np.array([w, h]) / 2.
    k = np.zeros(5)
    rt = cam['rt']
    t = cam['t']
    camera_mtx = np.array([[f[0], 0., c[0]],[0., f[1], c[1]],[0.,0.,1.]], dtype=np.float64)
    v = model.r.reshape((-1, 3)).copy()
    # camera_mtx_extmatrix = np.array([[1.0, 0.0, 0.0, t[0]], [0.0, 1.0, 0.0, t[1]],
    #                                  [0.0, 0.0, 1.0, t[2]]])
    # extra_col = np.ones(len(v))
    # v_new = np.c_[v, extra_col].T
    # uv = np.matmul(camera_mtx_extmatrix, v_new)
    # uv = np.matmul(camera_mtx, np.matmul(camera_mtx_extmatrix, v_new)).T
    # uv[:, 0] = uv[:, 0] / uv[:, 2]
    # uv[:, 1] = uv[:, 1] / uv[:, 2]
    uv1 = cv2.projectPoints(v, rt, t, camera_mtx, k)[0].squeeze()
    #test_3D_to_2D(uv1)
    uv_u = (uv1[:, 0] / float(image.shape[1])).reshape((len(uv1), 1))
    uv_v = (uv1[:, 1] / float(image.shape[0])).reshape((len(uv1), 1))
    uv = np.hstack((uv_u, uv_v))
    return (uv, v)

def generate_uv1(model, image, cam):
    model.betas[:len(cam['betas'])] = cam['betas']
    model.pose[:] = cam['pose']
    model.trans[:] = cam['trans']
    w, h = (image.shape[1], image.shape[0])
    f = np.array([w, w]) / 2. if cam['f'] is None else cam['f']
    c = np.array([w, h]) / 2.
    k = np.zeros(5)
    rt = cam['rt']
    t = cam['t']
    camera_mtx = np.array([[f[0], 0., c[0]],[0., f[1], c[1]],[0.,0.,1.]], dtype=np.float64)
    v = model.r.reshape((-1, 3)).copy()

    uv = cv2.projectPoints(v, rt, t, camera_mtx, k)[0].squeeze()
    for i in range(len(uv)):
        uv[i, 0] = int(uv[i, 0])
        uv[i, 1] = int(uv[i, 1])
    uv = uv.astype(int)
    return (uv, v)


def write_obj(path, model, verts, uvs):
    with open(path, 'w') as fp:
        fp.write("mtllib test.mtl\n")
        fp.write("\n")
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        fp.write("\n")
        for uv in uvs:
            fp.write('vt %f %f %f\n' % (uv[0], 1 - uv[1], 0.))
        fp.write("\n")
        fp.write("o m_avg\n")
        fp.write("g m_avg\n")
        fp.write("usemtl lambert1\n")
        fp.write("s 1\n")

        for face in model.f + 1:
            fp.write('f %d/%d %d/%d %d/%d\n' % (face[0], face[0], face[1],
                        face[1], face[2], face[2]))

def test_3D_to_2D(uvs1):
    import matplotlib.pyplot as plt
    img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/prediction_img_0330_vis.png")
    img = cv2.flip(img, 0)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.figure(1)
    plt.imshow(img)
    ax = plt.subplot(111)
    ax.scatter(uvs1[:,0], uvs1[:,1], c='r')
    #plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/demo_frame0012.png")
    plt.show()

def write_obj_and_translation(HR_img_path, HR_pkl_path, LR_pkl_path):

    ###########################HR#############################

    #image = cv2.imread("/home/lgh/unite the people/input/0small.jpg")
    image = cv2.imread(HR_img_path)     #HR
    model = _load_model("/home/lgh/code/SMPLify_TF/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")

    HR_pkl_files = _os.listdir(HR_pkl_path)
    HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                       key=lambda d: int(d.split('.')[0]))
    uvs1 = []
    for ind, HR_pkl_file in enumerate(HR_pkl_files):
        pkl_path = _os.path.join(HR_pkl_path, HR_pkl_file)
        with open(pkl_path) as f:
            cam = pickle.load(f)
        #focal = cam['f'] / 2   #2400.0
        #cam['f'] = cam['f'] / 2
        uvs1, verts1 = generate_uv(model, image, cam)
        np.savetxt(HR_pkl_path + "/uvs.txt", uvs1)
        write_obj(HR_pkl_path + "/0000.obj", model, verts1, uvs1)
        fi = open(HR_pkl_path + "/cameraHR.txt", "w")
        fi.write(str(cam['f'][0]) + " " + str(cam['t'][0]) + " " + str(cam['t'][1]) + " " + str(cam['t'][2]))
        fi.close()
    #################################LR##################################
    pkl_files = _os.listdir(LR_pkl_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                       key=lambda d: int(d.split('.')[0]))
    # output = LR_pkl_path
    # if not _os.path.exists(output):
    #     _os.makedirs(output)

    for ind, pkl_file in enumerate(pkl_files):
        pkl_path = _os.path.join(LR_pkl_path, pkl_file)
        with open(pkl_path) as f:
            cam = pickle.load(f)
        #cam['f'] = np.array([2500., 2500.])
        uvs, verts = generate_uv(model, image, cam)
        if ind == 2:
            a = verts
            b = 1
        write_obj(LR_pkl_path + "/%04d.obj" % ind, model, verts, uvs1)

        #save cam parameters
        f = open(LR_pkl_path + "/camera_%04d.txt" % ind, "w")
        f.write(str(cam['f'][0]) + " " + str(cam['t'][0]) + " " + str(cam['t'][1]) + " " + str(cam['t'][2]))
        f.close()


write_obj_and_translation(util.HR_img_base_path + "/aa1small.jpg", util.HR_img_base_path + "/output",
                "/home/lgh/code/SMPLify_TF/test/temp0/1/LR/output_358_440_replace")
