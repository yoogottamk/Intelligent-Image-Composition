"""
Grab Cut
"""

import cv2 as cv
import numpy as np
from friendblend.helpers import imshow


def grab_cut(img_l, img_r, bb_l, bb_r, fb_l, fb_r):

    fg_model = np.zeros((1,65), np.float64)
    bg_model = np.zeros((1,65), np.float64)


    mask = np.full((img_l.shape[0], img_l.shape[1]), cv.GC_PR_BGD ,np.uint8)
    bb_x, bb_y, bb_w, bb_h = bb_l
    fb_x, fb_y, fb_w, fb_h = fb_l



    # Label probable foreground and definite foreground using bounding box
    mask[bb_y:bb_y + bb_h, bb_x:bb_x + bb_w] = cv.GC_PR_FGD
    mask[fb_y:fb_y + fb_h, fb_x:fb_x + fb_w] = cv.GC_FGD

    mask, fg_model, bg_model = cv.grabCut(img_l, mask, None, bg_model, fg_model, 1, cv.GC_INIT_WITH_MASK)

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8') 
    img = img_l * mask[:,:,np.newaxis]   
    
    return img, img_r
