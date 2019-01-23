# coding:utf-8
import cv2

# 按照灰度图像的方式读入两幅图片
img1 = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small2/optimization_data/0004.jpg", cv2.IMREAD_GRAYSCALE)
for i in range(100):
    img2 = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/xiongfei/optimization_data/%04d.png" % i, cv2.IMREAD_GRAYSCALE)
    # 创建ORB特征检测器和描述符
    orb = cv2.ORB_create()
    # 对两幅图像检测特征和描述符
    keypoint1, descriptor1 = orb.detectAndCompute(img1, None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 利用匹配器 匹配两个描述符的相近成都
    maches = bf.match(descriptor1, descriptor2)
    # 按照相近程度 进行排序
    maches = sorted(maches, key=lambda x: x.distance)
    # 画出匹配项
    img3 = cv2.drawMatches(img1, keypoint1, img2, keypoint2, maches[: 5], img2, flags=2)
    cv2.imshow("matches", img3)
    cv2.waitKey(5000)
    a = 1