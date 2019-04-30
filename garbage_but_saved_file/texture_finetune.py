import os
import cv2
try:
    from smpl.serialization import load_model as _load_model
except:
    from smpl.smpl_webuser.serialization import load_model as _load_model
import util
import pickle
import numpy as np

def bgr_to_rgb(img):
    b, g, r = cv2.split(img)
    img_rgb = cv2.merge([r, g, b])
    return img_rgb

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
    uv[:, 1] = image.shape[0] - uv[:, 1]
    return (uv, v)

def generate_2D_texture(HR_img_path, LR_img_path, HR_pkl_path, LR_pkl_path):
    image = cv2.imread(HR_img_path)
    image = bgr_to_rgb(image)
    LR = cv2.imread(LR_img_path)
    LR = bgr_to_rgb(LR)
    model = _load_model(util.SMPL_PATH)
    with open(HR_pkl_path) as f:
        cam = pickle.load(f)
        # focal = cam['f'] / 2   #2400.0
        # cam['f'] = cam['f'] / 2
    uvs_HR, verts1 = generate_uv(model, image, cam)
    uvs_HR = np.around(uvs_HR).astype(np.int32)
    #################################LR##################################

    with open(LR_pkl_path) as f:
        cam = pickle.load(f)
        # cam['f'] = np.array([2500., 2500.])
    uvs_LR, verts = generate_uv(model, LR, cam)
    uvs_LR = np.around(uvs_LR).astype(np.int32)
    LR[uvs_LR[:, 1], uvs_LR[:, 0], :] = image[uvs_HR[:, 1], uvs_HR[:, 0], :]
    return LR

def compare_model_and_LR_in_2D():
    LR_img_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/LR/0290_2909_1681_318.jpg"
    HR_img_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/HR/aa1small.jpg"
    HR_pkl_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/HR/output/0000.pkl"
    LR_pkl_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/LR/output_290_440_use_in_final/0290.pkl"
    LR = generate_2D_texture(HR_img_path, LR_img_path, HR_pkl_path, LR_pkl_path)

    import matplotlib.pyplot as plt
    plt.imshow(LR)
    #fig = plt.figure(1)
    #ax = plt.subplot(111)
    #ax.scatter(LR[:, 0], LR[:, 1], c='b')
    #ax.scatter(HR_j2d[:, 0], HR_j2d[:, 1], c='r')
    #plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png")
    plt.show()

compare_model_and_LR_in_2D()
