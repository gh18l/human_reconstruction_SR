#coding=utf-8
import numpy as np
from matplotlib import pyplot as plt

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

def periodicDecomp(lr, hr, lr_points, lr_num, lr_len, hr_points, hr_num, hr_len):
    # 读取数据
    # lr_file = open('D:/TBSI/Lab2F/ICCP2019/pose/tianyi_pose_Jan/tianyi_pose/LR/tianyi_LR_pose.csv', 'rb')
    # hr_file = open('D:/TBSI/Lab2F/ICCP2019/pose/tianyi_pose_Jan/tianyi_pose/HR/tianyi_HR_pose.csv', 'rb')
    # lr_file = open(LRfile, 'rb')
    # hr_file = open(HRfile, 'rb')
    # lr = np.loadtxt(lr_file, delimiter=',', dtype=float)
    # hr = np.loadtxt(hr_file, delimiter=',', dtype=float)
    # lr_file.close()
    # hr_file.close()
    # lr_mean = np.mean(lr[0:90], axis=0)
    # hr_mean = np.mean(hr[6:82], axis=0)
    lr = lr
    hr = hr
    lr_mean = np.mean(lr[lr_points[0]:lr_points[-1]], axis=0)
    hr_mean = np.mean(hr[hr_points[0]:hr_points[-1]], axis=0)

    results = []
    for k in range(72):
        # 对HR分解周期并求和、求平均
        hr_4 = hr[:,k] #here
        # hr_pSeg = [6,21,36,51, 67,82]
        hr_pSeg = hr_points
        hr_4_s = []
        hr_segLen = []
        for p in range(hr_num):
            hr_4_s.append(hr_4[hr_pSeg[p]: (hr_pSeg[p+1]-1)])
            hr_segLen.append((hr_pSeg[p+1]-hr_pSeg[p])/lr_len)
        hr_part_mean = []
        for j in range(1,hr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(hr_num):
                tempSum += np.mean(hr_4_s[i][int(hr_segLen[i]*(j-1)):int(hr_segLen[i]*j)])
            hr_part_mean.append(tempSum/hr_num)
        hr_factor_mul_4 = np.array(hr_part_mean) / hr_mean[k]
        hr_factor_add_4 = np.array(hr_part_mean) - hr_mean[k]

        # 对LR分解周期并求和、求平均
        lr_4 = lr[:,k] # here
        # lr_pSeg = [0,13,31,47,61,75,90]
        lr_pSeg = lr_points
        lr_4_s = []
        lr_segLen = []
        for i in range(len(lr_pSeg) - 1):
            lr_4_s.append(lr_4[lr_pSeg[i]:(lr_pSeg[i+1]-1)])
            lr_segLen.append((lr_pSeg[i+1] - lr_pSeg[i])/lr_len)
        lr_part_mean = []
        for j in range(1,lr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(lr_num):
                tempSum += np.mean(lr_4_s[i][int(lr_segLen[i]*(j-1)):int(lr_segLen[i]*j)])
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

if __name__ == '__main__':
    # LRfile = 'D:/TBSI/Lab2F/ICCP2019/pose/tianyi_pose_Jan/tianyi_pose/LR/tianyi_LR_pose.csv' #LR文件输入
    # HRfile = 'D:/TBSI/Lab2F/ICCP2019/pose/tianyi_pose_Jan/tianyi_pose/HR/tianyi_HR_pose.csv' #HR文件输入
    lr = 
    hr =
    lr_points = [0,13,31,47,61,75,90] #LR周期点
    lr_num = 6 #LR周期数量（输入数据有几个周期）
    lr_len = 12 #LR最短周期长度（最短的周期有几帧）
    hr_points = [6,21,36,51, 67,82]
    hr_num = 5
    hr_len = 12

    output = periodicDecomp(lr, hr, lr_points, lr_num, lr_len, hr_points, hr_num, hr_len)