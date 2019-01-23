import os
import cv2
import numpy as np

base_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init"
pano_path = "/home/lgh/code/result.jpg"
render_people_list = ["dingjian", "xiongfei", "jianing", "zhicheng"]
start_times = [0, 0, 0, 0]
pano_to_ref_scale = 0.585

pano = cv2.imread(pano_path)

for ind, render_people in enumerate(render_people_list):
    coordination_file = os.path.join(base_path, render_people, "coordination.npy")
    coordination = np.load(coordination_file)
    render_img_path = os.path.join(base_path, render_people, "output")
    render_img_files = os.listdir(render_img_path)
    render_img_files = sorted([filename for filename in render_img_files if filename.endswith(".png") and "texture" in filename],
                        key=lambda d: int((d.split('_')[3]).split('.')[0]))
    start_time = start_times[ind]
    for ind, render_img_file in enumerate(render_img_files):
        img_file_path = os.path.join(render_img_path, render_img_file)
        render_img = cv2.imread(img_file_path)
        # TODO
        minification = 1.0
        ### downsample it
        render_img = cv2.resize(render_img,
                    (render_img.shape[1] * minification,
                        render_img.shape[0] * minification))



