import cv2
import numpy as np
import tensorflow as tf

def distance_transform(input):
    dst_type = cv2.DIST_L2
    output = cv2.distanceTransform(np.uint8(input), dst_type, 5)
    return output

#def distance_transform_tf(input):
