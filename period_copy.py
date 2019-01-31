#coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
# import util
# import smpl_np
# from opendr_render import render
import cv2
import csv
# import pandas as pd

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