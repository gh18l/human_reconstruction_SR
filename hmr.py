import numpy as np
import os

def get_original(proc_param, cam):
    img_size = proc_param['img_size']
    undo_scale = 1. / np.array(proc_param['scale'])

    cam_s = cam[0]
    cam_pos = cam[1:]
    flength = 500.
    tz = flength / (0.5 * img_size * cam_s)
    trans = np.hstack([cam_pos, tz])
    return trans

## trans3 pose72 beta10 =85
def get_hmr(hmr_init_path):
    hmr_init_files = os.listdir(hmr_init_path)
    hmr_init_files = sorted([filename for filename in hmr_init_files
                        if filename.endswith(".npy") and "theta" in filename],
                            key=lambda d: int((d.split('_')[1]).split('.')[0]))

    N = len(hmr_init_files)
    print("%d files" % N)
    hmr_tran = np.zeros([N, 3])
    hmr_theta = np.zeros([N, 72])
    hmr_beta = np.zeros([N, 10])
    hmr_cam = np.zeros([N, 3])
    hmr_joints3d = np.zeros([N, 19, 3])

    hmrcam_init_files = os.listdir(hmr_init_path)
    hmrcam_init_files = sorted([filename for filename in hmrcam_init_files
                             if filename.endswith(".npy") and "camera" in filename],
                            key=lambda d: int((d.split('_')[3]).split('.')[0]))
    hmrproc_init_files = os.listdir(hmr_init_path)
    hmrproc_init_files = sorted([filename for filename in hmrproc_init_files
                                if filename.endswith(".npy") and "proc" in filename],
                               key=lambda d: int((d.split('_')[2]).split('.')[0]))
    hmrjoints3d_init_files = os.listdir(hmr_init_path)
    hmrjoints3d_init_files = sorted([filename for filename in hmrjoints3d_init_files
                                 if filename.endswith(".npy") and "joints3d" in filename],
                                key=lambda d: int((d.split('_')[1]).split('.')[0]))
    for ind, hmr_init_file in enumerate(hmr_init_files):
        file = np.load(hmr_init_path + hmr_init_files[ind])
        file_cam = np.load(hmr_init_path + hmrcam_init_files[ind])
        file_proc = np.load(hmr_init_path + hmrproc_init_files[ind]).item()
        file_joints3d = np.load(hmr_init_path + hmrjoints3d_init_files[ind])
        trans = get_original(file_proc, file[0, 0:3])

        hmr_tran[ind, :] = trans
        hmr_theta[ind, :] = file[0, 3:75]
        hmr_beta[ind, :] = file[0, 75:85]
        hmr_cam[ind, :] = file_cam
        hmr_joints3d[ind, :, :] = file_joints3d.squeeze()
    return hmr_theta, hmr_beta, hmr_tran, hmr_cam, hmr_joints3d
