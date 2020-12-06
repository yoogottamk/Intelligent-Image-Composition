"""
Grab Cut
"""

from typing import Tuple

import cv2 as cv
import numpy as np

from friendblend.helpers import imshow
from friendblend.processing.helpers import connected


def grab_cut(img_l, img_r, fb_l, boundary=20):
    """
    Performs grabcut
     - extracts face from img_l using face bounding boxes
     - blends the two images
    """

    # Allocate memory for grabcut
    fg_model = np.zeros((1, 65), np.float64)
    bg_model = np.zeros((1, 65), np.float64)

    mask = np.full(img_l.shape[:2], cv.GC_PR_BGD, np.uint8)
    fb_x, fb_y, fb_w, fb_h = fb_l

    w = img_l.shape[1]

    # Label probable foreground and definite foreground using bounding box
    mask[
        fb_y + fb_h :,
        max(0, fb_x - boundary) : min(fb_x + fb_w + boundary, w),
    ] = cv.GC_PR_FGD
    mask[
        max(0, fb_y - boundary) : fb_y + fb_h,
        max(0, fb_x - boundary) : min(fb_x + fb_w + boundary, w),
    ] = cv.GC_FGD

    mask, _, _ = cv.grabCut(
        img_l, mask, None, bg_model, fg_model, 1, cv.GC_INIT_WITH_MASK
    )

    # Mask out image using mask obtained from grabcut
    mask = np.where(
        np.bitwise_or(mask == cv.GC_PR_BGD, mask == cv.GC_BGD), 0, 1
    ).astype("uint8")

    mask = filter_mask(mask, fb_l)

    img = img_l * mask[:, :, np.newaxis]
    # imshow(img)

    return img, crop_fg(img, img_r)


def filter_mask(mask: np.ndarray, fb: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Removes components which are not connected to the face bounding box from foreground
    """
    fx, fy, fw, fh = fb
    face_mid = (fy + (fh // 2), fx + (fw // 2))

    labels = connected(mask)

    return np.where(labels == labels[face_mid], 1, 0).astype("uint8")


def crop_fg(fg, bg):
    """
    Superimposes cropped foreground from grabcut onto background image
    """
    # performing erosion on binary fg image
    gray_fg = cv.cvtColor(fg, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray_fg, 0, 1, cv.THRESH_BINARY)

    erosion_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    out = cv.erode(mask, erosion_kernel, 5)

    # multiplying eroded mask with fg image to get final crop
    final_crop = fg * np.reshape(out, (out.shape[0], out.shape[1], 1))

    # overlaying fg on bg image to obtain final result
    out = 1 - out
    bg_merged = bg * np.reshape(out, (out.shape[0], out.shape[1], 1))
    bg_merged = bg_merged + final_crop

    return bg_merged
