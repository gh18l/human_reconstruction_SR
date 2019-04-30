import cv2
import os
import numpy as np

pano = cv2.imread("/home/lgh/code/result.jpg")
base_path = "/home/lgh/code/SMPLify_TF/test/temp0_12.10/"
LR_name_path = base_path + "result/input"
reneder_path = base_path + "result/input/render/"

LR1_files = os.listdir(LR_name_path)
LR1_files = sorted([filename for filename in LR1_files if filename.endswith(".jpg")],
						key=lambda d: int((d.split('_')[0])))
render_files = os.listdir(reneder_path)
render_files = sorted([filename for filename in render_files if filename.endswith(".png")],
						key=lambda d: int((d.split('_')[1]).split('.')[0]))

full_frame = 500
have_img_index = np.zeros([full_frame, 2])
for i, LR1_file in enumerate(LR1_files):
	x = int(LR1_file.split('_')[1])
	y = int(LR1_file.split('_')[2])
	t = int(LR1_file.split('_')[3].split('.')[0])
	have_img_index[t, 0] = x
	have_img_index[t, 1] = y

scale = 0.585
calibration_pano = np.array([1800.0, 2260.0])
calibration_ref = np.array([1064.0, 424.0])
calibration_ref_scaled = calibration_ref / scale
calibration_delta = calibration_pano - calibration_ref_scaled

render_ind = 0
for i in range(full_frame):
	pano_op = pano.copy()
	if have_img_index[i, 0] == 0 and have_img_index[i, 1] == 0:
		continue
		#pano_op = cv2.resize(pano_op, (pano_op.shape[1] / 4, pano_op.shape[0] / 4))
		#cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp0/result/%04d.jpg" % i, pano_op)
	else:
		render_in_LR_x = have_img_index[i, 0]
		render_in_LR_y = have_img_index[i, 1]
		render_in_LR_x_scaled = float(render_in_LR_x) / scale
		render_in_LR_y_scaled = float(render_in_LR_y) / scale
		render_in_pano_x = render_in_LR_x_scaled + calibration_delta[0]
		render_in_pano_y = render_in_LR_y_scaled + calibration_delta[1]
		render_img = cv2.imread(reneder_path + render_files[render_ind])
		render_ind = render_ind + 1

		render_img_w = int(600.0 / scale)
		render_img_h = int(450.0 / scale)

		render_img = cv2.resize(render_img, (render_img_w, render_img_h))
		for a in range(render_img.shape[0]):
			for b in range(render_img.shape[1]):
				if render_img[a, b, 0] != 0 and render_img[a, b, 1] != 0 and render_img[a, b, 2] != 0:
					pano_op[int(render_in_pano_y) + a, int(render_in_pano_x) + b, :] = render_img[a, b, :]

		#pano_op = cv2.resize(pano_op, (pano_op.shape[1] / 4, pano_op.shape[0] / 4))
		pano_op_crop = pano_op[(pano_op.shape[0] / 2):pano_op.shape[0],
					   		(pano_op.shape[1] / 2):pano_op.shape[1], :]
		cv2.imwrite(base_path + "result/%04d.jpg" % i, pano_op_crop)

