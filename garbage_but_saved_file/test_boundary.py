import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology, color, data, filters
def gradient():
    # image =color.rgb2gray(data.camera())
    image = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/HR/aa1small.jpg", 0)
    #denoised = filters.rank.median(image, morphology.disk(2))
    roi = cv2.selectROI("1", image)
    roi = np.array(roi)
    imCrop = image[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]
    markers = filters.rank.gradient(imCrop, morphology.disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = filters.rank.gradient(imCrop, morphology.disk(2))
    labels = morphology.watershed(gradient, markers, mask=imCrop)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axes = axes.ravel()
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(imCrop, cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title("Original")
    ax1.imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title("Gradient")
    ax2.imshow(markers, cmap=plt.cm.gray, interpolation='nearest')
    ax2.set_title("Markers")
    ax3.imshow(labels, cmap=plt.cm.gray, interpolation='nearest')
    ax3.set_title("Segmented")
    plt.show()
    # for ax in axes:
    #     ax.axis('off')
    #
    # fig.tight_layout()

gradient()
img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0/1/HR/aa1small.jpg")
roi = cv2.selectROI("1", img)
roi = np.array(roi)
imCrop = img[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]

# img = cv2.cvtColor(imCrop,cv2.COLOR_RGB2GRAY)
# lap = cv2.Laplacian(img,cv2.CV_64F)
# lap = np.uint8(np.absolute(lap))


imCrop_gray = cv2.cvtColor(imCrop,cv2.COLOR_RGB2GRAY)
x = cv2.Sobel(imCrop_gray, cv2.CV_16S, 1, 0)
y = cv2.Sobel(imCrop_gray, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
dst1 = np.zeros([dst.shape[0], dst.shape[1]])
for i in range(dst.shape[0]):  #row 1 2 3 ...
    index = np.argmax(dst[i,:])
    imCrop[i, index, :] = 255
img[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])] = imCrop
# gray = cv2.cvtColor(imCrop,cv2.COLOR_RGB2GRAY)
# xgrad=cv2.Sobel(gray,cv2.CV_16SC1,1,0)
# ygrad=cv2.Sobel(gray,cv2.CV_16SC1,0,1)
# edge_output=cv2.Canny(xgrad,ygrad,20,60)
cv2.imshow("edge",img)
cv2.waitKey()