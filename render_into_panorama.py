import os
import cv2
import numpy as np

base_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init"
pano_path = "/home/lgh/code/result.jpg"
output_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init/jianing2/final"
render_people_list = ["jianing2"]
    #["dingjian", "xiongfei", "jianing", "zhicheng"]
# start_time = [342, 647, 0, 0]
# end_time = [410, 727, 0, 0]
start_time = [0, 0, 1113, 0]
end_time = [0, 0, 1192, 0]
max_timestamp = 1200

pano_to_ref_scale = 0.585
local_to_ref_scale = 0.15
calibration_pano = np.array([1800.0, 2260.0])
calibration_ref = np.array([1064.0, 424.0])
### upsampling ref to equal size with pano
calibration_ref_scaled = calibration_ref / pano_to_ref_scale
### get size gap between pano and ref
calibration_delta = calibration_pano - calibration_ref_scaled

pano = cv2.imread(pano_path)
dict_prefix = {}
dict_suffix = {}
dict_coordination = {}
for ind, render_people in enumerate(render_people_list):
    coordination_file = os.path.join(base_path, render_people, "coordination.npy")
    coordination = np.load(coordination_file)
    render_img_path = os.path.join(base_path, render_people, "output_after_refine")
    render_img_files = os.listdir(render_img_path)
    render_img_files = sorted(
        [filename for filename in render_img_files if filename.endswith(".png") and "hmr_optimization_texture" in filename],
        key=lambda d: int((d.split('_')[3]).split('.')[0]))
    dict_prefix[render_people] = render_img_path
    dict_suffix[render_people] = render_img_files
    dict_coordination[render_people] = coordination

for t in range(max_timestamp):
    pano_temp = pano.copy()
    for ind, render_people in enumerate(render_people_list):
        coordination = dict_coordination[render_people]
        ### is it appeared in this timestamp?????
        if t < start_time[ind] or t > end_time[ind]:
            continue
        img_file_path = os.path.join(dict_prefix[render_people],
                dict_suffix[render_people][t-start_time[ind]])
        render_img = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)
        # TODO
        minification = 1.0
        ### downsample it
        render_img = cv2.resize(render_img,
                (int(render_img.shape[1] * minification),
                    int(render_img.shape[0] * minification)))
        x = coordination[t, 0]
        y = coordination[t, 1]
        ### convert input ref coordination to pano size
        render_x_scaled = float(x) / pano_to_ref_scale
        render_y_scaled = float(y) / pano_to_ref_scale
        ### calculate render position in panorama
        render_in_pano_x = render_x_scaled + calibration_delta[0]
        render_in_pano_y = render_y_scaled + calibration_delta[1]

        ### resize image into pano size, compatible with ref and origin texture
        render_img_w = int(600.0 / pano_to_ref_scale)
        render_img_h = int(450.0 / pano_to_ref_scale)
        render_img_scaled = cv2.resize(render_img, (render_img_w,
                                                    render_img_h))
        ### render pano size image into panorama
        for a in range(render_img_scaled.shape[0]):
            for b in range(render_img_scaled.shape[1]):
                if render_img_scaled[a, b, 0] > 0  and render_img_scaled[a, b, 1] > 0 and render_img_scaled[
                    a, b, 2] > 0:
                    pano_temp[int(render_in_pano_y) + a, int(render_in_pano_x) + b, :] = render_img_scaled[a, b, :]
    output_origin_path = os.path.join(output_path, "origin")
    # output_small_path = output_path + "/small"
    if not os.path.exists(output_origin_path):
        os.makedirs(output_origin_path)
    # if not os.path.exists(output_small_path):
    # os.makedirs(output_small_path)
    #pano_temp = cv2.resize(pano_temp, (pano_temp.shape[1] / 5,
                                       #pano_temp.shape[0] / 5))
    cv2.imwrite(os.path.join(output_origin_path, "pano_%04d.jpg" % t), pano_temp)




