#coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import util
import smpl_np
from opendr_render import render
import cv2
import csv
# import pandas as pd
import single_frame_estimation_hmr_LR_periodrefine as refine_opt
# 均值平滑
def mean_smoothing(s,r):
    s2=np.zeros(s.shape)
    len = s.size
    for i in range(r):
        temp1 = 0
        temp2 = 0
        for j in range(i):
            temp1 += s[i]
            temp2 += s[len - i -1]
        s2[i] = temp1 / (i+1)
        s2[len - i - 1] = temp2 / (i+1)
    for i in range(r, len - r):
        tempSum = 0
        for j in range(1, r+1):
            tempSum += (s[i-j] +s[i+j])
        s2[i]=(s[i]+tempSum) / (2*r + 1)
    return s2

# 指数平滑公式
def exponential_smoothing(s,alpha,r):
    s2=np.zeros(s.shape)
    len = s.size
    for i in range(r):
        s2[i] = s[i]
        s2[len - i - 1] = s[len - i -1]
    beta = (1-alpha) / (r*2)
    for i in range(r, len - r):
        tempSum = 0
        for j in range(1, r+1):
            tempSum += (s[i-j] +s[i+j])
        s2[i]=alpha*s[i]+beta*tempSum
    return s2

def periodicDecomp(lr, hr, lr_points, hr_points):
    lr = lr
    hr = hr
    lr_num = len(lr_points)-1
    hr_num = len(hr_points)-1
    lr_len = hr_len = 9999
    for i in range(lr_num):
        if lr_points[i+1] - lr_points[i] < lr_len:
            lr_len = lr_points[i+1] - lr_points[i]
    for i in range(hr_num):
        if hr_points[i+1] - hr_points[i] < hr_len:
            hr_len = hr_points[i+1] - hr_points[i]
    lr_mean = np.mean(lr[lr_points[0]:lr_points[-1]+1], axis=0)
    hr_mean = np.mean(hr[hr_points[0]:hr_points[-1]+1], axis=0)

    results = []
    for k in range(72):
        ### no change rot
        # if k < 3:
        #     results.append(np.array(lr[:, k][lr_points[0]:(lr_points[-1])]))
        #     continue
        # 对HR分解周期并求和、求平均
        hr_4 = hr[:,k] #here
        # hr_pSeg = [6,21,36,51, 67,82]
        hr_pSeg = hr_points
        hr_4_s = []
        hr_segLen = []
        for p in range(hr_num):
            hr_4_s.append(hr_4[hr_pSeg[p]: hr_pSeg[p+1]])
            hr_segLen.append((hr_pSeg[p+1]-hr_pSeg[p])/lr_len)
        hr_part_mean = []
        for j in range(1,hr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(hr_num):
                tempSum += np.mean(hr_4_s[i][int(hr_segLen[i]*(j-1)):int(hr_segLen[i]*j+1)])
            hr_part_mean.append(tempSum / hr_num)
        hr_factor_mul_4 = np.array(hr_part_mean) / hr_mean[k]
        hr_factor_add_4 = np.array(hr_part_mean) - hr_mean[k]

        # 对LR分解周期并求和、求平均
        lr_4 = lr[:,k] # here
        # lr_pSeg = [0,13,31,47,61,75,90]
        lr_pSeg = lr_points
        lr_4_s = []
        lr_segLen = []
        for i in range(len(lr_pSeg) - 1):
            lr_4_s.append(lr_4[lr_pSeg[i]:lr_pSeg[i+1]])
            lr_segLen.append((lr_pSeg[i+1] - lr_pSeg[i])/lr_len)
        lr_part_mean = []
        for j in range(1,lr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(lr_num):
                tempSum += np.mean(lr_4_s[i][int(lr_segLen[i]*(j-1)):int(lr_segLen[i]*j+1)])
            lr_part_mean.append(tempSum/lr_num)
        lr_factor_mul_4 = np.array(lr_part_mean) / lr_mean[k]
        lr_factor_add_4 = np.array(lr_part_mean) - lr_mean[k]
        # print(lr_mean[k])

        # 利用HR恢复LR-直接在LR均值上只用HR因子加法操作
        mline = np.ones([len(lr), 1]) *  lr_mean[k]
        lr_4_m = []
        for i in range(len(lr_pSeg) - 1):
            lr_4_m.append(mline[lr_pSeg[i]:lr_pSeg[i+1]])
        for j in range(1,lr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(lr_num):
                # print("--j:--",j,"--i:--",i)
                # print(lr_4_s[i][int(segLen[i]*(j-1)):int(segLen[i]*j)])
                lr_4_m[i][int(lr_segLen[i] * (j - 1)):int(lr_segLen[i] * j)] += hr_factor_add_4[j-1]
        result = []
        for i in range(len(lr_4_m)):
            for j in lr_4_m[i]:
                # print(j[0])
                result.append(j[0])
        results.append(np.array(result))
    output = np.array(results).T
    # data = pd.DataFrame(output)
    # data.to_csv('tianyi_pose_0111.csv',header = False, index = False) # here
    # data.to_csv(output_file,header = False, index = False) # here
    return output

def periodicCopy(lr, hr, lr_points, hr_points):
    results = []
    for k in range(72):
        lr_new = lr[:, k]
        hr_use = hr[:, k]
        lr_new[lr_points[0]:(lr_points[-1]+1)] = hr_use[hr_points[0]:(hr_points[-1]+1)]
        j = 0
        for i in range(lr_points[0]):
            index = hr_points[0] + j
            lr_new[i] = hr_use[index]
            if index > hr_points[-1]:
                j = 0
            j += 1
        j = 0
        for i in range(lr_points[-1]+1, len(lr_new)):
            index = hr_points[0] + j
            lr_new[i] = hr_use[index]
            if index > hr_points[-1]:
                j = 0
            j += 1
        results.append(np.array(lr_new))
    output = np.array(results).T
    return output

def save_prerefine_data(LR_cameras, texture_img, texture_vt, data_dict):
    if not os.path.exists(util.hmr_path + "refine_data"):
        os.makedirs(util.hmr_path + "refine_data")
    np.save(util.hmr_path + "refine_data/LR_cameras.npy", LR_cameras)
    cv2.imwrite(util.hmr_path + "refine_data/texture_img.jpg", texture_img)
    np.save(util.hmr_path + "refine_data/texture_vt.npy", texture_vt)
    np.save(util.hmr_path + "refine_data/data_dict.npy", data_dict)




def save_pkl_to_csv(pose_path):
    #####save csv before refine, extra output
    pkl_files = os.listdir(pose_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    length = len(pkl_files)
    array = np.zeros((length, 24 * 3))
    for ind, pkl_file in enumerate(pkl_files):
        pkl_path = os.path.join(pose_path, pkl_file)
        with open(pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        for i in range(24 * 3):
            array[ind, i] = pose[0, i]
    with open(os.path.join(pose_path, "optimization_pose.csv"), "w") as f:
        writer = csv.writer(f)
        for row in array:
            writer.writerow(row)

def refine_LR_pose(HR_pose_path, hr_points, lr_points):
    LR_cameras = np.load(util.hmr_path + "refine_data/LR_cameras.npy")
    texture_img = cv2.imread(util.hmr_path + "refine_data/texture_img.jpg")
    texture_vt = np.load(util.hmr_path + "refine_data/texture_vt.npy")
    data_dict = np.load(util.hmr_path + "refine_data/data_dict.npy").item()
    hmr_dict, _ = util.load_hmr_data(util.hmr_path)


    LR_path = util.hmr_path + "output"
    LR_pkl_files = os.listdir(LR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    LR_length = len(LR_pkl_files)
    LR_array = np.zeros((LR_length, 24 * 3))
    LR_betas = np.zeros([LR_length, 10])
    LR_trans = np.zeros([LR_length, 3])
    HR_path = HR_pose_path
    HR_pkl_files = os.listdir(HR_path)
    HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    HR_length = len(HR_pkl_files)
    HR_array = np.zeros((HR_length, 24 * 3))


    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        beta = param['betas']
        tran = param['trans']
        for i in range(24 * 3):
            LR_array[ind, i] = pose[0, i]
        for i in range(10):
            LR_betas[ind, i] = beta[0, i]
        for i in range(3):
            LR_trans[ind, i] = tran[0, i]
    for ind, HR_pkl_file in enumerate(HR_pkl_files):
        HR_pkl_path = os.path.join(HR_path, HR_pkl_file)
        with open(HR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        for i in range(24 * 3):
            HR_array[ind, i] = pose[0, i]

    output = periodicDecomp(LR_array, HR_array, lr_points, hr_points)
    refine_opt.refine_optimization(output, LR_betas, LR_trans, data_dict,
                            hmr_dict, LR_cameras, texture_img, texture_vt)

