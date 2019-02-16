import cv2
import numpy as np
import copy
flow = np.array([])
for i in range(100):
    img1 = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small2/optimization_data/%04d.jpg" % i)
    img2 = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small2/optimization_data/%04d.jpg" % (i+1))
    retval = cv2.DISOpticalFlow_create(2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #flow = np.zeros((450, 600, 2), dtype=np.float)
    #flow = np.array([])
    flow = retval.calc(img2, img1, flow)
    _flow = copy.deepcopy(flow)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            _flow[i, j, 0] =  j + flow[i, j, 0]
            _flow[i, j, 1] =  i + flow[i, j, 1]
    dst = cv2.remap(img1, _flow, np.array([]), cv2.INTER_CUBIC)
    cv2.imshow("dst", dst)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey()