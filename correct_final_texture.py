import cv2
import numpy as np
from sklearn import neighbors
import time
import copy
import util
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def stitch_in_picture(render, img):
    for i in range(render.shape[0]):
        for j in range(render.shape[1]):
            if render[i, j, 0] == 0 and render[i, j, 1] == 0 and render[i, j, 2] == 0:
                continue
            img[i, j, :] = render[i, j, :]
    return img

def erode_boundary(img, erodesize = 5):
    img_corrected = np.zeros_like(img)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erodesize, erodesize))
    eroded = cv2.erode(mask, kernel)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if eroded[i, j] == 255:
                img_corrected[i, j, :] = img[i, j, :]
    return img_corrected

def correct_dilate(img, dilatesize = 5):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilatesize, dilatesize))
    dilated = cv2.dilate(mask, kernel)

    ## extract eroded part
    border = dilated - mask

    nodes = []
    borders = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
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


def correct_render_mix(img):
    img_temp = copy.deepcopy(img)
    w = img.shape[1]
    h = img.shape[0]
    up_mid = util.correct_render_up_mid
    hand_up_mid = util.correct_render_hand_up_mid
    down_mid = util.correct_render_down_mid
    # img1 = np.zeros([mid, w, 3])
    # img2 = np.zeros([h - mid, w, 3])
    img1 = img_temp[0:up_mid, :, :]   #head
    img2 = img_temp[up_mid:hand_up_mid, :, :]  ## body
    img3 = img_temp[hand_up_mid:down_mid, :, :]  ## hands
    img4 = img_temp[down_mid:, :, :] ##leg
    #img_corrected1 = correct_render_small(img1, 0)
    img_corrected1 = img1
    img_corrected2 = correct_render_small(img2, 5)
    img_corrected3 = correct_render_small(img3, 5)
    img_corrected4 = correct_render_small(img4, 10)
    img_final = np.zeros_like(img)
    img_final[0:up_mid, :, :] = img_corrected1
    img_final[up_mid:hand_up_mid, :, :] = img_corrected2
    img_final[hand_up_mid:down_mid, :, :] = img_corrected3
    img_final[down_mid:, :, :] = img_corrected4

    #img_final_final = correct_dilate(img_final)
    #img_corrected1.copy(img_final[0:320, :, :])
    #img_corrected2.copy(img_final[320:, :, :])
    return img_final

def correct_render_mix_dilate(img):
    img_temp = copy.deepcopy(img)
    w = img.shape[1]
    h = img.shape[0]
    up_mid = util.correct_render_up_mid
    down_mid = util.correct_render_down_mid
    # img1 = np.zeros([mid, w, 3])
    # img2 = np.zeros([h - mid, w, 3])
    img1 = img_temp[0:up_mid, :, :]   #head
    img2 = img_temp[up_mid:down_mid, :, :]  ## body
    img3 = img_temp[down_mid:, :, :] ##leg
    img_corrected1 = correct_render_small(img1, 2)
    img_corrected2 = correct_render_small(img2, 5)
    img_corrected2 = correct_dilate(img_corrected2)
    img_corrected3 = correct_render_small(img3, 5)
    img_corrected3 = correct_dilate(img_corrected3)
    img_final = np.zeros_like(img)
    img_final[0:up_mid, :, :] = img_corrected1
    img_final[up_mid:down_mid, :, :] = img_corrected2
    img_final[down_mid:, :, :] = img_corrected3

    #img_final_final = correct_dilate(img_final)
    #img_corrected1.copy(img_final[0:320, :, :])
    #img_corrected2.copy(img_final[320:, :, :])
    return img_final


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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
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

def correct_render_small(img, erodesize = 5):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erodesize, erodesize))
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
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


