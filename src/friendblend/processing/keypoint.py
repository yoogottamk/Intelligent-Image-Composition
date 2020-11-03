"""
Contains code for keypoint detection and matching
"""

import cv2 as cv

from friendblend.processing.helpers import pt_in_box

__all__ = ["get_keypoints", "get_descriptors", "filter_keypoints"]


def get_keypoints(img, n_keypoints=1000):
    """
    Returns ORB keypoints
    """
    return cv.ORB_create(nfeatures=n_keypoints).detect(img)


def get_descriptors(img, kps, n_keypoints=1000):
    """
    Returns ORB descriptors
    """
    return cv.ORB_create(nfeatures=n_keypoints).compute(img, kps)[1]


def filter_keypoints(box1, box2, kps):
    """
    Only returns keypoints which are NOT in box1 or box2
    """
    return list(
        filter(
            lambda kp: not pt_in_box(kp.pt, box1) and not pt_in_box(kp.pt, box2),
            kps.copy(),
        )
    )
