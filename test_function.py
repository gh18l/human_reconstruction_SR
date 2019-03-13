import cv2
import numpy as np
import os
import pickle as pkl
import csv
from scipy import sparse as sp
import tensorflow as tf
def polyfit3D():
    import matplotlib.pyplot as plt
    path = "/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/LR/output"
    pkl_files = os.listdir(path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")])
    length = len(pkl_files)
    array = np.zeros((length, 24*3))
    index = 0
    for ind, pkl_file in enumerate(pkl_files):
        # if ind != 0 and ind % 2 == 0:
        #     continue
        pkl_path = os.path.join(path, pkl_file)
        with open(pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        for i in range(24 * 3):
            array[index, i] = pose[0, i]
        index = index + 1
    with open("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/LR/output/tianyi_LR_pose.csv", "w") as f:
        writer = csv.writer(f)
        for row in array:
            writer.writerow(row)
    for i in range(24*3):
        x = np.arange(1, length + 1, 1)
        y = array[:, i]
        reg = np.polyfit(x, y, 50)
        ry = np.polyval(reg, x)
        plt.plot(x, y, 'b^', label='groundtruth')
        plt.plot(x, ry, 'r.', label='regression')
        plt.legend(loc=0)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('%04d' % i)
        plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/HR/output/function_%04d.png" % i)
        plt.cla()
        #plt.show()
        #a = 1


def all_pose_output():
    params = []
    LR_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/LR/output"
    HR_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/HR/output"
    LR_pkl_files = os.listdir(LR_path)
    HR_pkl_files = os.listdir(HR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key = lambda d: int(d.split('.')[0]))
    HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int(d.split('.')[0]))
    LR_length = len(LR_pkl_files)
    HR_length = len(HR_pkl_files)
    LR_array = np.zeros((LR_length, 24 * 3))
    new_LR_array = np.zeros((LR_length, 24 * 3))
    HR_array = np.zeros((HR_length, 24 * 3))
    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
            params.append(param)
        pose = param['pose']
        for i in range(24 * 3):
            LR_array[ind, i] = pose[0, i]
    for ind, HR_pkl_file in enumerate(HR_pkl_files):
        HR_pkl_path = os.path.join(HR_path, HR_pkl_file)
        with open(HR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        for i in range(24 * 3):
            HR_array[ind, i] = pose[0, i]

    ######################2 3 5 6 8 9#####################
    # LR_array_clip = np.concatenate((LR_array[:,3:9], LR_array[:,12:18], LR_array[:,21:27]),
    #                                axis = 1)
    # HR_array_clip = np.concatenate((HR_array[:, 3:9], HR_array[:, 12:18], HR_array[:, 21:27]),
    #                                axis=1)
    # LR_array_clip = LR_array[:,12:18]
    # HR_array_clip = HR_array[:, 12:18]
    # LR_array_clip = np.concatenate((LR_array[:,12:13], LR_array[:,15:16],
    #             LR_array[:,3:4], LR_array[:,6:7]), axis=1)
    # HR_array_clip = np.concatenate((HR_array[:,12:13], HR_array[:,15:16],
    #             HR_array[:, 3:4], HR_array[:, 6:7]), axis=1)
    LR_array_clip = np.concatenate((LR_array[:, 12:13], LR_array[:, 15:16],
            LR_array[:, 3:4], LR_array[:, 6:7], LR_array[:, 1:2]), axis=1)
    HR_array_clip = np.concatenate((HR_array[:, 12:13], HR_array[:, 15:16],
            HR_array[:, 3:4], HR_array[:, 6:7], HR_array[:, 1:2]), axis=1)
    # LR_array_clip = np.concatenate((LR_array[:, 3:4], LR_array[:, 6:7]), axis=1)
    # HR_array_clip = np.concatenate((HR_array[:, 3:4], HR_array[:, 6:7]), axis=1)
    patch_size = 1
    for i in range(0, len(LR_array) / patch_size * patch_size, patch_size):
        min_value = 999999.0
        best_index = 0
        for j in range(len(HR_array) - patch_size + 1):
            error_matrix = LR_array_clip[i:i+patch_size,:] - HR_array_clip[j:j+patch_size,:]
            error_value = sum(sum(np.fabs(error_matrix)))
            if error_value < min_value:
                min_value = error_value
                best_index = j
        new_LR_array[i:i+patch_size, :] = HR_array[best_index:best_index+patch_size, :]
        for patch_ind in range(patch_size):
            params[i + patch_ind]['pose'] = HR_array[best_index + patch_ind, :]
            with open("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/HR_replace_LR_output/%04d.pkl" % (i + patch_ind), 'w') as outf:
                pickle.dump(params[i + patch_ind], outf)

def generate_new_pkl():
    with open("/home/lgh/code/SMPLify_TF/test/temp0/1/HR_prediction/output/HR_ouput_120804.CSV", "r") as f:
        reader = csv.reader(f)
        array = []
        for row in reader:
            print(row)
            array.append(row)

    array = array[1:35]
    LR_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/HR_GT/output"
    LR_pkl_files = os.listdir(LR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int(d.split('.')[0]))
    LR_length = len(LR_pkl_files)
    LR_array = np.zeros((LR_length, 24 * 3))
    params = []
    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        params.append(param)
        pose = param['pose']
        for i in range(24 * 3):
            LR_array[ind, i] = pose[0, i]
    start_frame = 36
    for ind in range(34):
        str_func = lambda x: x.replace('\xef\xbb\xbf', '')
        #print(len(array[ind][0]))
        LR_array[ind, 3] = float(str_func(array[ind][1]))
        params[ind]['pose'][0, 3] = LR_array[ind, 3]
        LR_array[ind, 4] = float(str_func(array[ind][2]))
        params[ind]['pose'][0, 4] = LR_array[ind, 4]
        LR_array[ind, 5] = float(str_func(array[ind][3]))
        params[ind]['pose'][0, 5] = LR_array[ind, 5]
        LR_array[ind, 12] = float(str_func(array[ind][4]))
        params[ind]['pose'][0, 12] = LR_array[ind, 12]
        LR_array[ind, 13] = float(str_func(array[ind][5]))
        params[ind]['pose'][0, 13] = LR_array[ind, 13]
        LR_array[ind, 14] = float(str_func(array[ind][6]))
        params[ind]['pose'][0, 14] = LR_array[ind, 14]
        with open("/home/lgh/code/SMPLify_TF/test/temp0/1/HR_GT/output/replace_3451213114_compare/%04d.pkl" % ind, 'wb') as fout:
            pickle.dump(params[ind], fout)
    a = 1

def watch_pose_value():
    LR_path = "/home/lgh/code/SMPLify_TF/test/temp0/2/LR/output"
    LR_pkl_files = os.listdir(LR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int(d.split('.')[0]))
    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        print("%d" % ind, pose[0, 48:51], pose[0, 51:54])
        a = 1

def HR_pose_prediction_full_replace_LR_pose():
    with open("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/LR_of_using_HR_great/tianyi_pose_0111.csv", "r") as f:
        reader = csv.reader(f)
        array = []
        for row in reader:
            print(row)
            array.append(row)
    LR_path = "/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/LR/output"
    LR_pkl_files = os.listdir(LR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int(d.split('.')[0]))
    LR_length = len(LR_pkl_files)
    LR_array = np.zeros((LR_length, 24 * 3))
    params = []
    str_func = lambda x: x.replace('\xef\xbb\xbf', '')
    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        if ind < 90:
            LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
            with open(LR_pkl_path) as f:
                param = pickle.load(f)
            params.append(param)
            pose = param['pose']
            for i in range(72):
                params[ind]['pose'][0, i] = float(str_func(array[ind][i]))
            with open("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/LR_of_using_HR_great/%04d.pkl" % ind, 'wb') as fout:
                pickle.dump(params[ind], fout)


    a = 1

def HR_pose_replace_LR_pose():
    with open("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/HR_multi/output/HR_tianyi_original_predict_forwardto84.csv", "r") as f:
        reader = csv.reader(f)
        array = []
        for row in reader:
            print(row)
            array.append(row)
    #array = array[85:106]
    array = array[1:17]
    LR_path = "/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/LR/output_0_100"
    LR_pkl_files = os.listdir(LR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int(d.split('.')[0]))
    LR_length = len(LR_pkl_files)
    LR_array = np.zeros((LR_length, 24 * 3))
    params = []
    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        params.append(param)
        pose = param['pose']
        for i in range(24 * 3):
            LR_array[ind, i] = pose[0, i]
    for ind in range(len(array)):
        if ind <= 16:
            str_func = lambda x: x.replace('\xef\xbb\xbf', '')
            #print(len(array[ind][0]))
            LR_array[ind + 84, 3] = float(str_func(array[ind][4]))
            params[ind + 84]['pose'][0, 3] = LR_array[ind + 84, 3]
            LR_array[ind + 84, 4] = float(str_func(array[ind][5]))
            params[ind + 84]['pose'][0, 4] = LR_array[ind + 84, 4]
            LR_array[ind + 84, 5] = float(str_func(array[ind][6]))
            params[ind + 84]['pose'][0, 5] = LR_array[ind + 84, 5]
            LR_array[ind + 84, 6] = float(str_func(array[ind][7]))
            params[ind + 84]['pose'][0, 6] = LR_array[ind + 84, 6]
            LR_array[ind + 84, 7] = float(str_func(array[ind][8]))
            params[ind + 84]['pose'][0, 7] = LR_array[ind + 84, 7]
            LR_array[ind + 84, 8] = float(str_func(array[ind][9]))
            params[ind + 84]['pose'][0, 8] = LR_array[ind + 84, 8]

            LR_array[ind + 84, 12] = float(str_func(array[ind][13]))
            params[ind + 84]['pose'][0, 12] = LR_array[ind + 84, 12]
            LR_array[ind + 84, 13] = float(str_func(array[ind][14]))
            params[ind + 84]['pose'][0, 13] = LR_array[ind + 84, 13]
            LR_array[ind + 84, 14] = float(str_func(array[ind][15]))
            params[ind + 84]['pose'][0, 14] = LR_array[ind + 84, 14]
            LR_array[ind + 84, 15] = float(str_func(array[ind][16]))
            params[ind + 84]['pose'][0, 15] = LR_array[ind + 84, 15]
            LR_array[ind + 84, 16] = float(str_func(array[ind][17]))
            params[ind + 84]['pose'][0, 16] = LR_array[ind + 84, 16]
            LR_array[ind + 84, 17] = float(str_func(array[ind][18]))
            params[ind + 84]['pose'][0, 17] = LR_array[ind + 84, 17]


            # LR_array[ind, 21] = float(str_func(array[ind][21]))
            # params[ind]['pose'][0, 21] = LR_array[ind, 21]
            # LR_array[ind, 22] = float(str_func(array[ind][22]))
            # params[ind]['pose'][0, 22] = LR_array[ind, 22]
            # LR_array[ind, 23] = float(str_func(array[ind][23]))
            # params[ind]['pose'][0, 23] = LR_array[ind, 23]
            # LR_array[ind, 24] = float(str_func(array[ind][24]))
            # params[ind]['pose'][0, 24] = LR_array[ind, 24]
            # LR_array[ind, 25] = float(str_func(array[ind][25]))
            # params[ind]['pose'][0, 25] = LR_array[ind, 25]
            # LR_array[ind, 26] = float(str_func(array[ind][26]))
            # params[ind]['pose'][0, 26] = LR_array[ind, 26]
            # LR_array[ind, 54] = float(str_func(array[ind][4]))
            # params[ind]['pose'][0, 54] = LR_array[ind, 54]
            # LR_array[ind, 57] = float(str_func(array[ind][5]))
            # params[ind]['pose'][0, 57] = LR_array[ind, 57]
            with open("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/LR_of_using_HR_pose_replace_shijianxuliefenxi_0_100/%04d.pkl" % ind, 'wb') as fout:
                pickle.dump(params[ind + 84], fout)
    a = 1

def new_and_old_mask_compare():
    new_mask_path = "/home/lgh/Downloads/Mask_RCNN-master/results"
    new_mask_files = os.listdir(new_mask_path)
    new_mask_files = sorted([filename for filename in new_mask_files if filename.endswith(".png")],
                          key=lambda d: int(d.split('_')[1].split('.')[0]))

    img_path = "/home/lgh/Downloads/Mask_RCNN-master/images"
    img_files = os.listdir(img_path)
    img_files = sorted([filename for filename in img_files if filename.endswith(".jpg")],
                            key=lambda d: int(d.split('_')[0]))

    old_mask_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/LR"
    old_mask_files = os.listdir(old_mask_path)
    old_mask_files = sorted([filename for filename in old_mask_files if filename.endswith(".png") and "mask" in filename],
                       key=lambda d: int((d.split('_')[1]).split('.')[0]))

    for ind, new_mask_file in enumerate(new_mask_files):
        new_mask_img = cv2.imread(os.path.join(new_mask_path, new_mask_file))
        img = cv2.imread(os.path.join(img_path, img_files[ind]))
        old_mask_img = cv2.imread(os.path.join(old_mask_path, old_mask_files[ind]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if old_mask_img[i, j, 0] != 0 and old_mask_img[i, j, 1] != 0 and old_mask_img[i, j, 2] != 0:
                    old_mask_img[i, j, :] = img[i, j, :]
                if new_mask_img[i, j, 0] != 0 and new_mask_img[i, j, 1] != 0 and new_mask_img[i, j, 2] != 0:
                    new_mask_img[i, j, :] = img[i, j, :]
        cv2.imwrite("/home/lgh/code/SMPLify_TF/test/new_and_old_mask_compare/%04d_old.png" % ind, old_mask_img)
        cv2.imwrite("/home/lgh/code/SMPLify_TF/test/new_and_old_mask_compare/%04d_new.png" % ind, new_mask_img)

def test_mask():
    path = "/media/lgh/T100/test_hmr_init/dingjianLR/"
    img = cv2.imread(path + "0300_2906_1737_328.jpg")
    mask_img = cv2.imread(path + "mask_0300.png")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask_img[i, j, 0] != 0 and mask_img[i, j, 1] != 0 and mask_img[i, j, 2] != 0:
                mask_img[i, j, :] = img[i, j, :]
    cv2.imwrite(path + "compare_300.png", mask_img)

def preprocessHR():
    path = "/home/lgh/code/SMPLify_TF/test/temp_test_VNect/"
    j2d_files = os.listdir(path)
    j2d_files = sorted([filename for filename in j2d_files if filename.endswith(".jpg")])
    for ind, j2d_file in enumerate(j2d_files):
        img = cv2.imread(path + j2d_file)
        roi = cv2.selectROI("1", img)  ## x y w h
        roi = np.array(roi)
        crop_height = roi[3]
        crop_width = roi[3]
        crop_x = int(roi[0])-(crop_height/2-int(roi[0])/2)
        if crop_x < 0:
            crop_x=0
        imCrop = img[int(roi[1]): int(roi[1] + roi[3]), crop_x:crop_x+crop_width]
        img_output = cv2.resize(imCrop, (180, 180), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp_test_VNect_crop/%04d.jpg" % ind, img_output)
    # for i in range(5):
    #     ret, frame = cap.read()
    #     crop_size = (1000, 750)
    #     img_new = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
    #     # img = cv2.imread("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/HR_multi/%04d.jpg"%i)
    #     roi = cv2.selectROI("1", img_new)   ## x y w h
    #     roi = np.array(roi)
    #     roi = roi * 4
    #     crop_height = roi[3]
    #     crop_width = roi[3] * 4 / 3
    #
    #     imCrop = frame[int(roi[1]) : int(roi[1] + roi[3]), int(roi[0])-1100:int(roi[0])-1100+crop_width]
    #     img_output = cv2.resize(imCrop, (600,450), interpolation=cv2.INTER_CUBIC)
    #     cv2.imwrite("/home/lgh/code/SMPLify_TF/test/temp0_tianyi/1/HR_resize/%04d.jpg" % i, img_output)

def mask_texture():
    for i in range(39):
        img = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small/optimization_data/%04d.jpg" % (i * 2))
        mask = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small/optimization_data/mask_%04d.png" % i)
        result = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask[:,:,0])
        result_weights = cv2.addWeighted(result,0.5,img,0.5,0)
        cv2.imwrite(
            "/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small/output_mask/mask_%04d.png" % i,
            result)
        cv2.imwrite("/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small/output_mask/compare_%04d.png" % i, result_weights)
        #cv2.imwrite(
            #"/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small3/optimization_data/mask_%04d.png" % i,
            #mask)

def generate_video():
    fps = 15
    size = (600, 450)
    path = "/home/lgh/code/SMPLify_TF/test/temp0/1/LR"
    video_path = "/home/lgh/code/SMPLify_TF/test/temp0/1/LR/LR.avi"
    videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    imgs_files = os.listdir(path)
    imgs_files = sorted([filename for filename in imgs_files if filename.endswith(".jpg")],
                        key=lambda d: int((d.split('_')[0])))
    for ind, imgs_file in enumerate(imgs_files):
        img_file_path = os.path.join(path, imgs_file)
        HR_img = cv2.imread(img_file_path)
        videowriter.write(HR_img)

def generate_coordination():
    path = "/home/lgh/code/SMPLify_TF/test/temp0/4/LR/img_data"
    img_files = os.listdir(path)
    img_files = sorted([filename for filename in img_files if filename.endswith(".jpg") and "mask" not in filename],
                       key=lambda d: int((d.split('_')[0])))
    f1 = open(path + '/coordination.txt', 'w')
    for i in range(230, 320):
        img_file = img_files[i]
        x = int(img_file.split('_')[1])
        y = int(img_file.split('_')[2])
        f1.write(str(x)+' '+str(y)+'\n')
generate_coordination()
#generate_video()
#mask_texture()
#HR_pose_prediction_full_replace_LR_pose()
#preprocessHR()
#HR_pose_prediction_full_replace_LR_pose()
#test_mask()
#new_and_old_mask_compare()
#all_pose_output()
#flipped_rot = cv2.Rodrigues(np.array([0.48754, -0.1316, -0.1498]))[0]
#HR_pose_replace_LR_pose()
#generate_new_pkl()
#watch_pose_value()
#polyfit3D()
# a = np.zeros([10, 2])
# b = np.zeros([10, 10])
# a = a.astype(np.int32)
# b[a[:, 0], a[:, 1]] = 1
#
# dst_type = cv2.cv.CV_DIST_L2 if cv2.__version__[0] == '2' else cv2.DIST_L2
#
# mask = cv2.imread("/home/lgh/Downloads/Mask_RCNN-master/results/mask_0018.png")
# dist_i = cv2.distanceTransform(np.uint8(mask[:, :, 0]), dst_type, 5) - 1
# dist_i[dist_i < 0] = 0
# dist_i[dist_i > 50] = 50
# dist_o = cv2.distanceTransform(255 - np.uint8(mask[:, :, 0]), dst_type, 5)
# dist_o[dist_o > 50] = 50
# cv2.imwrite("/home/lgh/Downloads/Mask_RCNN-master/results/mask_0001_transform.png", dist_i)
# cv2.imwrite("/home/lgh/Downloads/Mask_RCNN-master/results/masko_0001_transform.png", dist_o)