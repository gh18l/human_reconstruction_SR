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

    #######################SMPL is left-hand coordination, we convert it into right-hand coordination##################
    #v[:, 2] = -v[:, 2]
    ###################################################################################

    uv = cv2.projectPoints(v, rt, t, camera_mtx, k)[0].squeeze()
    #test_3D_to_2D(uv)
    uv_u = (uv[:, 0] / float(image.shape[1])).reshape((len(uv), 1))
    uv_v = (uv[:, 1] / float(image.shape[0])).reshape((len(uv), 1))
    uv = np.hstack((uv_u, uv_v))
    return (uv, v)

# def generate_uv(image, cam, verts):
#     w, h = (image.shape[1], image.shape[0])
#     f = np.array([w, w]) / 2. if cam['f'] is None else cam['f']
#     c = np.array([w, h]) / 2.
#     k = np.zeros(5)
#     rt = cam['rt']
#     t = cam['t']
#     camera_mtx = np.array([[f[0], 0., c[0]],[0., f[1], c[1]],[0.,0.,1.]], dtype=np.float64)
#
#     #######################SMPL is left-hand coordination, we convert it into right-hand coordination##################
#     #v[:, 2] = -v[:, 2]
#     ###################################################################################
#     uv = cv2.projectPoints(verts, rt, t, camera_mtx, k)[0].squeeze()
#     #test_3D_to_2D(uv)
#     uv_u = (uv[:, 0] / float(image.shape[1])).reshape((len(uv), 1))
#     uv_v = (uv[:, 1] / float(image.shape[0])).reshape((len(uv), 1))
#     uv = np.hstack((uv_u, uv_v))
#     return uv

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
            fp.write('vt %f %f %f\n' % (uv[0], uv[1], 0.))
        fp.write("\n")
        fp.write("o m_avg\n")
        fp.write("g m_avg\n")
        fp.write("usemtl lambert1\n")
        fp.write("s 1\n")

        for face in model.f + 1:
            fp.write('f %d/%d %d/%d %d/%d\n' % (face[0], face[0], face[1],
                        face[1], face[2], face[2]))

def test_3D_to_2D(uvs):
    import matplotlib.pyplot as plt
    img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/prediction_img_0322_vis.png")
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.figure(1)
    plt.imshow(img)
    ax = plt.subplot(111)
    ax.scatter(uvs[:,0], uvs[:,1])
    #plt.savefig("/home/lgh/code/SMPLify/smplify_public/code/temp/HRtemp/aa1_%04d.png" % o)
    plt.show()

def write_HR_obj_and_translation(HR_img, HR_pkl, uv, model):
    if uv.size() != 0:
        uvs1 = []
        uvs1, verts1 = generate_uv(model, HR_img, HR_pkl)
        #np.savetxt(util.HR_pkl_base_path + "/uvs.txt", uvs1)
        write_obj(util.HR_pkl_base_path + "/0000.obj", model, verts1, uvs1)
        fi = open(util.HR_pkl_base_path + "/cameraHR.txt", "w")
        fi.write(str(HR_pkl['f'][0]) + " " + str(HR_pkl['t'][0]) + " " + str(HR_pkl['t'][1]) + " " + str(HR_pkl['t'][2]))
        fi.close()
        return uvs1
    else:
        return uv

def write_LR_obj_and_translation(LR_img, LR_pkl, HR_uv, model, save_index):
    # cam['f'] = np.array([2500., 2500.])
    uvs, verts = generate_uv(model, LR_img, LR_pkl)
    write_obj(util.LR_pkl_base_path + "/%04d.obj" % save_index, model, verts, HR_uv)

    # save cam parameters
    f = open(util.LR_pkl_base_path + "/camera_%04d.txt" % save_index, "w")
    f.write(str(LR_pkl['f'][0]) + " " + str(LR_pkl['t'][0]) + " " + str(LR_pkl['t'][1]) + " " + str(LR_pkl['t'][2]))
    f.close()

def write_obj_and_translation(HR_img_path, HR_pkl_path, LR_pkl_path):

    ###########################HR#############################

    #image = cv2.imread("/home/lgh/unite the people/input/0small.jpg")
    image = cv2.imread(HR_img_path)     #HR
    model = _load_model(util.SMPL_PATH)

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
        #######################temp#########################
        #cam['pose'][0, 1] = -0.2
        ######################################################
        #cam['f'] = np.array([2500., 2500.])
        uvs, verts = generate_uv(model, image, cam)
        write_obj(LR_pkl_path + "/%04d.obj" % ind, model, verts, uvs1)

        #save cam parameters
        f = open(LR_pkl_path + "/camera_%04d.txt" % ind, "w")
        f.write(str(cam['f'][0]) + " " + str(cam['t'][0]) + " " + str(cam['t'][1]) + " " + str(cam['t'][2]))
        f.close()

def write_obj_and_translation1(HR_img_path, HR_pkl_path, LR_pkl_path):

    ###########################HR#############################

    #image = cv2.imread("/home/lgh/unite the people/input/0small.jpg")
    HR_img_files = _os.listdir(HR_img_path)
    HR_img_files = sorted([filename for filename in HR_img_files if filename.endswith(".pkl")],
                          key=lambda d: int(d.split('.')[0]))
    image = cv2.imread(HR_img_path)     #HR
    model = _load_model(util.SMPL_PATH)

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
        write_obj(LR_pkl_path + "/%04d.obj" % ind, model, verts, uvs1)

        #save cam parameters
        f = open(LR_pkl_path + "/camera_%04d.txt" % ind, "w")
        f.write(str(cam['f'][0]) + " " + str(cam['t'][0]) + " " + str(cam['t'][1]) + " " + str(cam['t'][2]))
        f.close()

def npz_to_obj(HR_img_path, HR_pkl_path, LR_pkl_path):
    path = "/home/lgh/code/SMPLify_TF/test/temp0/test_hmr_xiongfei_compare_optimization/output_verts"

    image = cv2.imread(HR_img_path)  # HR
    model = _load_model(util.SMPL_PATH)

    HR_pkl_files = _os.listdir(HR_pkl_path)
    HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int(d.split('.')[0]))
    uvs1 = []
    for ind, HR_pkl_file in enumerate(HR_pkl_files):
        pkl_path = _os.path.join(HR_pkl_path, HR_pkl_file)
        with open(pkl_path) as f:
            cam = pickle.load(f)
        # focal = cam['f'] / 2   #2400.0
        # cam['f'] = cam['f'] / 2
        uvs1, verts1 = generate_uv(model, image, cam)
        np.savetxt(HR_pkl_path + "/uvs.txt", uvs1)
        write_obj(HR_pkl_path + "/0000.obj", model, verts1, uvs1)
        fi = open(HR_pkl_path + "/cameraHR.txt", "w")
        fi.write(str(cam['f'][0]) + " " + str(cam['t'][0]) + " " + str(cam['t'][1]) + " " + str(cam['t'][2]))
        fi.close()
    #################################LR##################################
    npy_files = _os.listdir(path)
    npy_files = sorted([filename for filename in npy_files if filename.endswith(".npy") and "camera" not in filename],
                       key=lambda d: int((d.split('.')[0])))
    camera_files = _os.listdir(path)
    camera_files = sorted([filename for filename in camera_files if filename.endswith(".npy") and "camera" in filename],
                       key=lambda d: int((d.split('_')[1]).split('.')[0]), )
    for ind, npy_file in enumerate(npy_files):
        npy_file_path = _os.path.join(path, npy_file)
        camera_file_path = _os.path.join(path, camera_files[ind])
        npy = np.load(npy_file_path)
        camera = np.load(camera_file_path)
        verts = npy.squeeze()
        cam = camera.squeeze()
        write_obj(LR_pkl_path + "/%04d.obj" % ind, model, verts, uvs1)
        npy = np.load(npy_file_path)
        verts = npy.squeeze()
        # save cam parameters
        f = open(LR_pkl_path + "/camera_%04d.txt" % ind, "w")
        f.write("500.0" + " " + str(cam[0]) + " " + str(cam[1]) + " " + str(cam[2]))
        f.close()


# def write_obj_and_translation(HR_img_path, HR_pkl_path, LR_pkl_path, HR_verts, LR_verts):
#
#     ###########################HR#############################
#
#     #image = cv2.imread("/home/lgh/unite the people/input/0small.jpg")
#     image = cv2.imread(HR_img_path)     #HR
#     model = _load_model("/home/lgh/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
#
#     HR_pkl_files = _os.listdir(HR_pkl_path)
#     HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
#                        key=lambda d: int(d.split('.')[0]))
#     uvs1 = []
#     for ind, HR_pkl_file in enumerate(HR_pkl_files):
#         pkl_path = _os.path.join(HR_pkl_path, HR_pkl_file)
#         with open(pkl_path) as f:
#             cam = pickle.load(f)
#         #focal = cam['f'] / 2   #2400.0
#         #cam['f'] = cam['f'] / 2
#         uvs1 = generate_uv(image, cam, HR_verts[ind])
#         np.savetxt(HR_pkl_path + "/uvs.txt", uvs1)
#         write_obj(HR_pkl_path + "/0000.obj", model, HR_verts[ind], uvs1)
#         fi = open(HR_pkl_path + "/cameraHR.txt", "w")
#         fi.write(str(cam['f'][0]) + " " + str(cam['t'][0]) + " " + str(cam['t'][1]) + " " + str(cam['t'][2]))
#         fi.close()
#     #################################LR##################################
#     pkl_files = _os.listdir(LR_pkl_path)
#     pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
#                        key=lambda d: int(d.split('.')[0]))
#     # output = LR_pkl_path
#     # if not _os.path.exists(output):
#     #     _os.makedirs(output)
#     for ind, pkl_file in enumerate(pkl_files):
#         pkl_path = _os.path.join(LR_pkl_path, pkl_file)
#         with open(pkl_path) as f:
#             cam = pickle.load(f)
#         write_obj(LR_pkl_path + "/%04d.obj" % ind, model, LR_verts[ind], uvs1)
#
#         #save cam parameters
#         f = open(LR_pkl_path + "/camera_%04d.txt" % ind, "w")
#         f.write(str(cam['f'][0]) + " " + str(cam['t'][0]) + " " + str(cam['t'][1]) + " " + str(cam['t'][2]))
#         f.close()

#write_obj_and_translation(util.HR_img_base_path + "/aa1small.jpg", util.HR_img_base_path + "/output",
                #util.LR_img_base_path + "/output")
#npz_to_obj(util.HR_img_base_path + "/aa1small.jpg", util.HR_img_base_path + "/output",
                #"/home/lgh/code/SMPLify_TF/test/temp0/test_hmr_xiongfei_compare_optimization/output_verts")