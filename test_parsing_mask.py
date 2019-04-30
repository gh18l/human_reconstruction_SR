import cv2

img = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/output_nonrigid/test.png")
mask = img[:, :, 0]
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for ind in range(len(contours)):
    for i in range(len(contours[ind])):
        x = contours[ind][i, :, 0]
        y = contours[ind][i, :, 1]
        mask[y, x] = 255
cv2.imshow("1", mask)
cv2.waitKey()
