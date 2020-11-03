"""
The FriendBlend pipeline
"""

import logging
import sys

import cv2 as cv
import numpy as np
from rich.logging import RichHandler

from friendblend.processing.color_correction import apply_clahe
from friendblend.processing.face_body_detection import get_bounds
from friendblend.processing.keypoint import (
    get_keypoints,
    get_descriptors,
    filter_keypoints,
)
from friendblend.processing import helpers as processing_helpers
from friendblend.helpers import log_all_methods, log_all_in_module
from friendblend import global_vars

from friendblend import processing

log_all_in_module(processing.color_correction)
log_all_in_module(processing.face_body_detection)
log_all_in_module(processing.keypoint)


@log_all_methods()
class Blend:
    """
    Blend two images with friends into one
    """

    def __init__(self, img1_path: str, img2_path: str):
        self.log = logging.getLogger()
        self.img1 = self.imload(img1_path, ensure_success=True)
        self.img2 = self.imload(img2_path, ensure_success=True)

    def imload(
        self, img_path, mode: int = 1, ensure_success: bool = False
    ) -> np.ndarray:
        """
        Loads an image
        Dies if ensure_success is true and image read error occurs
        """
        im = cv.imread(img_path, mode)

        if im is None:
            self.log.error("Couldn't load image at '%s'", img_path)

            if ensure_success:
                self.log.error("ensure_success set, dying")
                sys.exit(1)

        return im

    @staticmethod
    def color_correction(img, clip_limit=3.0, n_bins=256, grid=(7, 7)):
        """
        Color correction using CLAHE
        """
        return apply_clahe(img, clip_limit=clip_limit, n_bins=n_bins, grid=grid)

    @staticmethod
    def get_face_body_bounds(img):
        """
        Calculates face and body bounds
        """
        return get_bounds(img)

    @staticmethod
    def get_valid_keypoints(img1, img2, bb1, bb2):
        """
        Calculates and filters ORB descriptors
        """
        kps1 = get_keypoints(img1)
        kps2 = get_keypoints(img2)

        filtered_kps1 = filter_keypoints(bb1, bb2, kps1)
        filtered_kps2 = filter_keypoints(bb1, bb2, kps2)

        ds1 = get_descriptors(img1, filtered_kps1)
        ds2 = get_descriptors(img2, filtered_kps2)

        return filtered_kps1, ds1, filtered_kps2, ds2

    def blend(self):
        """
        Performs the FriendBlend algorithm
        """
        cc1 = Blend.color_correction(self.img1)
        cc2 = Blend.color_correction(self.img2)

        fb1, bb1 = Blend.get_face_body_bounds(self.img1)
        fb2, bb2 = Blend.get_face_body_bounds(self.img2)

        boxed1 = processing_helpers.draw_box(cc1, fb1)
        boxed1 = processing_helpers.draw_box(boxed1, bb1)

        boxed2 = processing_helpers.draw_box(cc2, fb2)
        boxed2 = processing_helpers.draw_box(boxed2, bb2)

        kps1, ds1, kps2, ds2 = Blend.get_valid_keypoints(cc1, cc2, bb1, bb2)

        return kps1, ds1, kps2, ds2


if __name__ == "__main__":
    global_vars.initialize()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
    )
    img_path = "../misc/images/fullbody.jpg"
    blend = Blend(img_path, img_path)

    blend.blend()
