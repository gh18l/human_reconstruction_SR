import cv2
import os
import pickle

def gradient(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradxy

def render_into_pano():
    # base_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/LR/output"
    # pkl_files = os.listdir(base_path)
    # pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
    #                    key=lambda d: int(d.split('.')[0]))
    # for ind, pkl_file in enumerate(pkl_files):
    #     pkl_path = os.path.join(base_path, pkl_file)
    #     with open(pkl_path) as f:
    #         cam = pickle.load(f)
    #     a = 1
    for i in range(100):
        img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/output/result_%04d.jpg" % i)
        scale = 0.55

render_into_pano()
# result = cv2.imread("/home/lgh/code/result.jpg")
#
# imgs = []
# render_imgs = []
# for i in range(100):
#     img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/%04d.jpg" % i)
#     render_img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/output/result_%04d.jpg" % i)
#     imgs.append(img)
#     render_imgs.append(render_img)
#
# max_value = 0
# max_localtion = []
# result_temp = []
# scale_good = []
# for i in range(20):
#     scale = 0.5 + i * 0.01
#     height, width = result.shape[:2]
#     size = (int(width * scale), int(height * scale))
#     result_temp = cv2.resize(result, size)
#     result_temp_grad = gradient(result_temp)
#     img_grad = gradient(imgs[i])
#     res = cv2.matchTemplate(result_temp, imgs[i], cv2.TM_CCOEFF_NORMED)
#     res_grad = cv2.matchTemplate(result_temp_grad, img_grad, cv2.TM_CCOEFF_NORMED)
#     ref_final = res *  res_grad
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ref_final)
#     if max_val > max_value:
#         max_value = max_val
#         max_localtion = max_loc
#         scale_good = scale
# top_left = max_localtion
# h = imgs[0].shape[0]
# w = imgs[0].shape[1]
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(result_temp, top_left, bottom_right, 255, 2)
# print(scale_good)
# height, width = result_temp.shape[:2]
# size = (int(width * 0.2), int(height * 0.2))
# result_temp = cv2.resize(result_temp, size)
# cv2.imshow("1",result_temp)
# cv2.waitKey()
