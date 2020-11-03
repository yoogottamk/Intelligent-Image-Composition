"""
Common functions used multiple times
"""

import cv2 as cv


def draw_box(img, bounding_box):
    """
    Draws a bounding box in image
    """
    img = img.copy()
    x, y, w, h = bounding_box

    return cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)


def pt_in_box(pt, box):
    """
    pt: (x1, y1)
    box: (x, y, w, h)

    returns whether point is within box
    """
    x1, y1 = pt
    x, y, w, h = box
    return x <= x1 <= x + w and y <= y1 <= y + h
