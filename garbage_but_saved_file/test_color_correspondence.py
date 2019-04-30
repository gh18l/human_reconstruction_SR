import cv2
import numpy as np
img1 = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/HR_multi_img1/0009.jpg")
img2 = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/0317_2858_1812_345.jpg")

roi = cv2.selectROI("1", img1)
roi = np.array(roi)
imCrop = img1[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]
res = cv2.matchTemplate(img2, imCrop, cv2.TM_CCOEFF_NORMED)
loc = cv2.minMaxLoc(res)

cv2.rectangle(img2, loc[3], (loc[3][0]+roi[2], loc[3][1]+roi[3]), (0,0,255))
cv2.imshow("2", img2)
cv2.waitKey()

a =1


