import cv2
import numpy as np

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    eroded = cv2.erode(mask, kernel)

    ## extract eroded part
    border = mask - eroded

    nodes = []
    for i in range(eroded.shape[0]):
        for j in range(eroded.shape[1]):
            if eroded[i, j] == 255:
                nodes.append([j, i])
    nodes = np.array(nodes)

    print("1111")

    ## replace color
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if border[i, j] == 255:
                min_index = closest_node(np.array([j, i]), nodes)
                ### replace color
                x = nodes[min_index, 0]
                y = nodes[min_index, 1]
                img_corrected[i, j, :] = img[y, x, :]
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
    for i in range(eroded.shape[0]):
        for j in range(eroded.shape[1]):
            if eroded[i, j] == 255:
                nodes.append([j, i])
    nodes = np.array(nodes)

    ## replace color
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if border[i, j] == 255:
                min_index = closest_node(np.array([j, i]), nodes)
                ### replace color
                x = nodes[min_index, 0]
                y = nodes[min_index, 1]
                img_corrected[i, j, :] = img[y, x, :]
    return img_corrected

def correct_rendertest(img):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    eroded = cv2.erode(mask, kernel)

    ## extract eroded part
    border = mask - eroded

    ## replace color
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if border[i, j] == 255:
                min_dis = 999999999.0
                min_index = [0, 0]
                for k in range(i - 15, i + 15):
                    for l in range(j - 15, j + 15):
                        if eroded[k, l] == 255:
                            dis = np.square(i - k) + np.square(j - l)
                            if dis < min_dis:
                                min_dis = dis
                                min_index = [k, l]
                ### replace color
                img_corrected[i, j, :] = img[min_index[0], min_index[1], :]
    return img_corrected
# img = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/jianing2/output_after_refine_big/hmr_optimization_texture_0012.png")
# img_corrected = correct_render(img)
# cv2.imshow("1", img_corrected)
# cv2.waitKey()


