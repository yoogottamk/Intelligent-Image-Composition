"""
Alpha Blending
"""

import cv2 as cv
import numpy as np

def alpha_blend(img_l, img_r, bb_l, bb_r):
    
    # defining variables described in the paper
    col_start = bb_l[0] + bb_l[2]
    col_end = bb_r[0]
    step_size = 1./(col_end - col_start)

    # initializing output
    res = img_l.copy()

    # running algorithm described in the paper
    for x in range(col_start, col_end+1):
        step_count = x - col_start
        res[:,x,0] = (1 - step_count*step_size)*img_l[:,x,0] + \
                     (step_count*step_size)*img_r[:,x,0]
        res[:,x,1] = (1 - step_count*step_size)*img_l[:,x,1] + \
                     (step_count*step_size)*img_r[:,x,1]
        res[:,x,2] = (1 - step_count*step_size)*img_l[:,x,2] + \
                     (step_count*step_size)*img_r[:,x,2]
    res[:,col_end+1:,:] = img_r[:,col_end+1:,:]

    return res
