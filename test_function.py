import cv2
import numpy as np
import os
import pickle as pkl
import csv
from scipy import sparse as sp
from skimage import morphology
import correct_final_texture as tex
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

# def generate_video():
#     fps = 15
#     size = (1920, 1080)
#     path = "/home/lgh/MOTdatasets/MOT17-08/img1/optimization_data"
#     video_path = "/home/lgh/MOTdatasets/MOT17-08/img1/optimization_data/1.mp4"
#     videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
#     imgs_files = os.listdir(path)
#     imgs_files = sorted([filename for filename in imgs_files if filename.endswith(".jpg")],
#                         key=lambda d: int((d.split('.')[0])))
#     for ind, imgs_file in enumerate(imgs_files):
#         img_file_path = os.path.join(path, imgs_file)
#         HR_img = cv2.imread(img_file_path)
#         videowriter.write(HR_img)

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

def crop_img():
    path = "/home/lgh/MOTdatasets/MOT17-08/img1/optimization_data"
    f1 = open(path + '/coordination.txt', 'w')
    for i in range(71):
        img1 = cv2.imread("/home/lgh/MOTdatasets/MOT17-08/img1/optimization_data/%06d.jpg" % (i+290))
        roi = cv2.selectROI("1", img1)
        roi = np.array(roi)
        roi[3] = 600
        roi[2] = 450
        imCrop = img1[int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]
        f1.write(str(int(roi[1])) + ' ' + str(int(roi[1] + roi[3])) + str(int(roi[0])) + ' ' + str(int(roi[0] + roi[2])) + '\n')
        cv2.imwrite("/home/lgh/MOTdatasets/MOT17-08/img1/optimization_data/%06d.jpg" % i, imCrop)
#generate_coordination()
# for i in range(6):
#     img = cv2.imread("/home/lgh/code/SMPLify_TF/test/test_suzhuo/optimization_data/%04d.png" % (i+51))
#     img = cv2.resize(img, (1000, 750))
#     cv2.imwrite("/home/lgh/code/SMPLify_TF/test/test_suzhuo/optimization_data/%04d.png" % i, img)
#crop_img()

def render_texture_imgbg():
    path = "/home/lgh/MPIIdatasets/img16"
    LR = cv2.imread(path + "_resize/optimization_data/0000.png")
    fps = 10
    size = (LR.shape[1] * 2, LR.shape[0] * 2)
    video_path = path + "/result.mp4"
    videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    for ind in range(40):
        HR = cv2.imread(path + "/optimization_data/%08d.jpg" % (ind+34))
        LR = cv2.imread(path + "_resize/optimization_data/%04d.png" % ind)
        render = cv2.imread(path + "_resize/output/hmr_optimization_texture_%04d.png" % ind)
        SR = cv2.imread("/home/lgh/MPIIdatasets/sr_result/pack6/%04d.png" % ind)
        SR = LR
        bg_img = np.copy(SR)
        for i in range(render.shape[0]):
            for j in range(render.shape[1]):
                if render[i, j, 0] == 0 and render[i, j, 1] == 0 and render[i, j, 2] == 0:
                    continue
                bg_img[i, j, :] = render[i, j, :]
        cv2.imwrite(path + "/all/HR_%04d.png" % ind, HR)
        cv2.imwrite(path + "/all/LR_%04d.png" % ind, LR)
        cv2.imwrite(path + "/all/render_%04d.png" % ind, bg_img)
        cv2.imwrite(path + "/all/SR_%04d.png" % ind, SR)
        result = np.zeros([LR.shape[0] * 2, LR.shape[1] * 2, 3],dtype=np.uint8)
        result[:LR.shape[0], :LR.shape[1], :] = LR
        result[:LR.shape[0], LR.shape[1]:, :] = SR
        result[LR.shape[0]:, :LR.shape[1], :] = bg_img
        result[LR.shape[0]:, LR.shape[1]:, :] = HR
        videowriter.write(result)

import matplotlib.pyplot as plt

def get_texture(solution = 32):
    #
    #inputs:
    #   solution is the size of generated texture, in notebook provided by facebookresearch the solution is 200
    #   If use lager solution, the texture will be sparser and smaller solution result in denser texture.
    #   im is original image
    #   IUV is densepose result of im
    #output:
    #   TextureIm, the 24 part texture of im according to IUV
    solution_float = float(solution) - 1

    IUV = cv2.imread("/home/lgh/Downloads/DensePose-master/DensePoseData/output_data/0028_IUV.png")
    im = cv2.imread("/home/lgh/Downloads/DensePose-master/DensePoseData/input_data/0028.png")
    index = IUV[:, :, 0]
    U = IUV[:,:,1]
    V = IUV[:,:,2]
    parts = list()
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        actual_part = np.zeros((solution, solution, 3))
        x,y = np.where(IUV[:,:,0]==PartInd)
        if len(x) == 0:
            parts.append(actual_part)
            continue


        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        tex_map_coords = ((255-v_current_points)*solution_float/255.).astype(int),(u_current_points*solution_float/255.).astype(int)
        for c in range(3):
            actual_part[tex_map_coords[0],tex_map_coords[1], c] = im[x,y,c]
        parts.append(actual_part)


    TextureIm  = np.zeros([solution*6,solution*4,3])

    for i in range(4):
        for j in range(6):
            TextureIm[ (solution*j):(solution*j+solution)  , (solution*i):(solution*i+solution) ,: ] = parts[i*6+j]



    cv2.imshow("1", TextureIm.transpose([1,0,2])[:,:,::-1]/255)
    cv2.waitKey()
    #plt.figure(figsize = (25,25))

    #plt.imshow(TextureIm.transpose([1,0,2])[:,:,::-1]/255)

def dilate_mask():
    img = cv2.imread("/media/lgh/6626-63BC/neet_to_mask/0030.png")
    mask = cv2.imread("/media/lgh/000324E9000CDBD5/mask_refine/yinheng_output_0015_49.ppm")
    mask = mask[:, :, 0]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                mask[i, j] = 1
    mask = mask.astype(np.bool)
    dst = morphology.remove_small_objects(mask, min_size=300, connectivity=1)
    mask = (dst.astype(np.uint8)) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.dilate(mask, kernel)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contour = contours[-1]
    # demo = np.zeros_like(mask)
    # for i in range(len(contour)):
    #     contour = contour.squeeze()
    #     x = contour[i, 0]
    #     y = contour[i, 1]
    #     demo[y, x] = 127
    result = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    result_weights = cv2.addWeighted(result, 0.5, img, 0.5, 0)
    cv2.imshow("1", result)
    cv2.waitKey()
   # cv2.imwrite("/home/lgh/real_system_data/data1/real_system_maskrcnn_for_mask_3.26/3/mask_dilated_0008.png", result_weights)
   # cv2.imwrite("/home/lgh/real_system_data/data1/real_system_maskrcnn_for_mask_3.26/3/maskdilatedresult_0008.png",
               # mask)

def delete_redundant_frame():
    path = "/home/lgh/real_system_data2/data1/people9/LR"
    index = 0
    f2 = open(path + "1/coordination.txt", 'w')
    with open("/home/lgh/real_system_data2/data1/people9/LR/coordination.txt", 'r') as f:
        for i in range(481):
            img_path = os.path.join(path, "%04d.png"%i)
            img = cv2.imread(img_path)
            line = f.readline()
            if i == 0 or i %2 == 1:
                cv2.imwrite("/home/lgh/real_system_data2/data1/people9/LR1/%04d.png" % index, img)
                a = []
                for j in line.split():
                    a.append(int(j))
                f2.write(str(a[0]) + " " + str(a[1]) + " " + str(a[2]) + "\n")
                index = index + 1

def view_mask():
    img_path = "/home/lgh/Downloads/DensePose-master/DensePoseData/demo_data4/"
    for i in range(32):
        img = cv2.imread(img_path + "img_%04d.jpg")
        mask = cv2.imread("/home/lgh/Downloads/DensePose-master/DensePoseData/infer_out1/img_0001_IUV.png")

def convert_mask(img, mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0:
                continue
            mask[i, j, :] = 255
    mask = mask[:, :, 0]
    return mask

def reverse_mask(mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                mask[i, j] = 255
            else:
                mask[i, j] = 0
    return mask
def mask_img(img, mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                img[i, j, :] = 255
    return img

def generate_video():
    path = "/home/lgh/MPIIdatasets/img5_resize8/output"
    for i in range(41):
        img = cv2.imread(path + "/texture_bg_%04d.png" % i)
        if i == 0:
            fps = 10
            size = (img.shape[1], img.shape[0])
            video_path = path + "/texture.mp4"
            videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
        videowriter.write(img)

def video_to_imgs():
    frame_count = 0
    cap = cv2.VideoCapture("/home/lgh/real_system_data/data1/human_render/rendercase1.mp4")
    success = True
    while(success):
        success, frame = cap.read()
        cv2.imwrite("/home/lgh/real_system_data/data1/human_render/rendercase1/%d.png" % frame_count, frame)
        frame_count = frame_count + 1

def resize_img():
    for i in range(28):
        img = cv2.imread("/home/lgh/real_system_data7/data1/people1/HR/optimization_data/%04d.jpg" % i)
        img_re = cv2.resize(img, (750, 1100))
        cv2.imwrite("/home/lgh/real_system_data7/data1/people1/HR1/%04d.jpg" % i, img_re)

def split_video():
    frame_count = 0
    cap = cv2.VideoCapture("/home/lgh/real_system_data7/data1/local1.avi")
    success = True
    while (success):
        success, frame = cap.read()
        frame1 = frame[:, :2100, :]
        frame2 = frame[:, 1500:, :]
        cv2.imwrite("/home/lgh/real_system_data7/data1/HR1/%d.png" % frame_count, frame1)
        cv2.imwrite("/home/lgh/real_system_data7/data1/HR2/%d.png" % frame_count, frame2)
        frame_count = frame_count + 1
split_video()
#resize_img()
#video_to_imgs()
#generate_video()
#generate_video()
# for i in range(27):
#     img = cv2.imread("/home/lgh/MPIIdatasets/img10/optimization_data/%08d.jpg" % (i+31))
#     img_resize = cv2.resize(img, (1280/4, 720/4))
#     img_resize = cv2.resize(img_resize, (1280, 720))
#     cv2.imwrite("/home/lgh/MPIIdatasets/img10_resize/optimization_data/%04d.png" % i, img_resize)
#
# img = cv2.imread("/home/lgh/MPIIdatasets/img16/optimization_data/00000055.jpg")
# mask = cv2.imread("/home/lgh/MPIIdatasets/img16/optimization_data/label.png")
# mask = convert_mask(img, mask)
# result = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8),
#                  mask=mask)
#result = tex.dilute_texture(img, mask, rect_size = 5)
#borders_region = reverse_mask(borders_region)
# img = mask_img(img, borders_region)
# img = cv2.resize(img, (256, 256))
# borders_region = cv2.resize(borders_region, (256, 256))
# cv2.imwrite("/home/lgh/mask.png", borders_region)
# cv2.imwrite("/home/lgh/img.png", img)

# result_weights = cv2.add(result, np.zeros(np.shape(result), dtype=np.uint8),
#                  mask=eroded)
# result_weights = cv2.addWeighted(result_weights, 0.5, result, 0.5, 0)
# cv2.imwrite("/home/lgh/MPIIdatasets/img16/output_nonrigid/texture.png", result)
# cv2.imshow("1", result)
# cv2.waitKey()
# img = cv2.imread("/home/lgh/real_system_data5/data1/people9/HR/optimization_data/0040.jpg")
# cv2.imshow("1", img)
# cv2.waitKey()
#delete_redundant_frame()
# img = cv2.imread("/home/lgh/Downloads/Mask_RCNN-master/images/00000078.jpg")
# mask = cv2.imread("/home/lgh/Downloads/Mask_RCNN-master/results/mask_0000.png")
# #cv2.imwrite("/home/lgh/Downloads/Mask_RCNN-master/images/00000078.jpg", img)
# result = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask[:, :, 0])
# #result_weights = cv2.addWeighted(result, 0.5, img, 0.5, 0)
# cv2.imshow("1", result)
# cv2.waitKey()
#dilate_mask()
#get_texture()
#render_texture_imgbg()
# img = cv2.imread("/home/lgh/MPIIdatasets/img13/output_nonrigid/hmr_optimization_texture_nonrigid0032.png")
# img = img[:, img.shape[1] / 3:, :]
# cv2.imshow("1", img)
# cv2.waitKey()

# for i in range(900):
#     img = cv2.imread("/home/lgh/MOTdatasets/MOT17-11-SDP/img1/optimization_data/%06d.jpg" % (i+1))
#     img_crop = img[:, 0:img.shape[1] / 2, :]
#     cv2.imwrite("/home/lgh/MOTdatasets/MOT17-11-SDP/img1/optimization_data/%06d.jpg" % i, img_crop)

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