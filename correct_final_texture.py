import cv2
import numpy as np
from sklearn import neighbors
import time
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def correct_render_big(img):
    img_corrected = np.copy(img)
    masks = np.zeros_like(img)
    mask = masks[:, :, 0]
    ## extract mask
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 255

    ## erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    eroded = cv2.erode(mask, kernel)

    ## extract eroded part
    border = mask - eroded

    nodes = []
    borders = []
    for i in range(eroded.shape[0]):
        for j in range(eroded.shape[1]):
            if eroded[i, j] == 255:
                nodes.append([j, i])
            if border[i, j] == 255:
                borders.append([j, i])
    nodes = np.array(nodes)
    borders = np.array(borders)
    labels = np.array(range(nodes.shape[0]))
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(nodes, labels)
    predictedLabel = knn.predict(borders)
    # img_corrected[borders[:, 0], borders[:, 1], :] = img[nodes[predictedLabel, 0], nodes[predictedLabel, 1], :]
    for i in range(borders.shape[0]):
        x = borders[i, 0]
        y = borders[i, 1]
        index = predictedLabel[i]
        x_nearest = nodes[index, 0]
        y_nearest = nodes[index, 1]
        img_corrected[y, x, :] = img[y_nearest, x_nearest, :]
    return img_corrected

def correct_render_small(img):
    img_corrected = np.copy(img)
    masks = np.zeros_like(img)
    mask = masks[:, :, 0]
    ## extract mask
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 255

    ## erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(mask, kernel)

    ## extract eroded part
    border = mask - eroded

    nodes = []
    borders = []
    for i in range(eroded.shape[0]):
        for j in range(eroded.shape[1]):
            if eroded[i, j] == 255:
                nodes.append([j, i])
            if border[i, j] == 255:
                borders.append([j, i])
    nodes = np.array(nodes)
    borders = np.array(borders)
    labels = np.array(range(nodes.shape[0]))
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(nodes, labels)
    predictedLabel = knn.predict(borders)
    # img_corrected[borders[:, 0], borders[:, 1], :] = img[nodes[predictedLabel, 0], nodes[predictedLabel, 1], :]
    for i in range(borders.shape[0]):
        x = borders[i, 0]
        y = borders[i, 1]
        index = predictedLabel[i]
        x_nearest = nodes[index, 0]
        y_nearest = nodes[index, 1]
        img_corrected[y, x, :] = img[y_nearest, x_nearest, :]
    return img_corrected

def correct_rendertest(img):
    img_corrected = np.copy(img)
    masks = np.zeros_like(img)
    mask = masks[:, :, 0]
    start = time.time()
    ## extract mask
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 255

    ## erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    eroded = cv2.erode(mask, kernel)

    ## extract eroded part
    border = mask - eroded

    nodes = []
    borders = []
    for i in range(eroded.shape[0]):
        for j in range(eroded.shape[1]):
            if eroded[i, j] == 255:
                nodes.append([j, i])
            if border[i, j] == 255:
                borders.append([j, i])
    nodes = np.array(nodes)
    borders = np.array(borders)
    labels = np.array(range(nodes.shape[0]))
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(nodes, labels)
    predictedLabel = knn.predict(borders)
    #img_corrected[borders[:, 0], borders[:, 1], :] = img[nodes[predictedLabel, 0], nodes[predictedLabel, 1], :]
    for i in range(borders.shape[0]):
        x = borders[i, 0]
        y = borders[i, 1]
        index = predictedLabel[i]
        x_nearest = nodes[index, 0]
        y_nearest = nodes[index, 1]
        img_corrected[y, x, :] = img[y_nearest, x_nearest, :]
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    return img_corrected
# img = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/jianing2/output_after_refine_big/hmr_optimization_texture_0012.png")
# img_corrected = correct_rendertest(img)
# cv2.imshow("1", img_corrected)
# cv2.waitKey()


