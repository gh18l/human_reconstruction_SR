import os
import numpy as np
def img_coordination_to_file(path):
    imgs_files = os.listdir(path)
    imgs_files = sorted([filename for filename in imgs_files if filename.endswith(".jpg")],
                        key=lambda d: int((d.split('_')[0])))
    array = np.zeros([10000, 2])
    for ind, imgs_file in enumerate(imgs_files):
        t = int(imgs_file.split('_')[3].split('.')[0])
        array[t, 0] = int(imgs_file.split('_')[1])
        array[t, 1] = int(imgs_file.split('_')[2])
    np.save(path + "/coordination.npy", array)

img_coordination_to_file("/home/lgh/code/SMPLify_TF/test/temp0/4/LR/img_data")
