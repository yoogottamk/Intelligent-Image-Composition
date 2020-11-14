"""
Grab Cut
"""

import cv2 as cv
import numpy as np

from friendblend.helpers import imshow


def grab_cut(img_l, img_r, fb_l, boundary=15):
    """
    Performs grabcut
     - extracts face from img_l using face bounding boxes
     - blends the two images
    """
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

    mask, fg_model, bg_model = cv.grabCut(
        img_l, mask, None, bg_model, fg_model, 1, cv.GC_INIT_WITH_MASK
    )

    mask = np.where(
        np.bitwise_or(mask == cv.GC_PR_BGD, mask == cv.GC_BGD), 0, 1
    ).astype("uint8")
    img = img_l * mask[:, :, np.newaxis]

    # imshow(img)

    return crop_fg(img, img_r)


def crop_fg(fg, bg):
    # TODO: move this line elsewhere in pipeline to handle image sizes
    bg = bg[
        bg.shape[0] - fg.shape[0] : bg.shape[0],
        bg.shape[1] - fg.shape[1] : bg.shape[1],
        :,
    ]

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
