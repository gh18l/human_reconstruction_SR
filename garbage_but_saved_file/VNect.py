import numpy as np
import cv2
def draw(v, img, option):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if option == "2d":
        fig = plt.figure(1)
        plt.imshow(img)
        plt.scatter(v[:, 0], v[:, 1], c='r')
        plt.show()
    if option == "3d":
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c='b', s=100)
        plt.show()


path_2d = "/home/lgh/code/SMPLify_TF/test/temp_test_VNect_crop/vnect(1)/test_2d.txt"
data_2d = np.loadtxt(path_2d)
data_2d = data_2d.reshape(-1,2)
data_3d = np.loadtxt(path_3d)
data_3d = data_3d.reshape(-1,3)
for i in range(20):
    img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp_test_VNect_crop/%04d.jpg" % i)
    draw(data_3d[i*21:i*21+20, :], img, "3d")

