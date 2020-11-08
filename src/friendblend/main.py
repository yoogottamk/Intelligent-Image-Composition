"""
The FriendBlend pipeline
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool

import cv2 as cv
import numpy as np
from rich.logging import RichHandler

from friendblend.processing.color_correction import apply_clahe
from friendblend.processing.face_body_detection import get_bounds
from friendblend.processing.keypoint import ORB, filter_keypoints, find_homography
from friendblend.processing import helpers as processing_helpers
from friendblend.helpers import log_all_methods, log_all_in_module, imshow
from friendblend import global_vars

from friendblend import processing

log_all_in_module(processing.color_correction)
log_all_in_module(processing.face_body_detection)
log_all_in_module(processing.keypoint)


def _process_blend(img):
    """
    only functions at module level are pickle-able
    multiprocessing involves pickling stuff so this had to be a top-level function
    """
    return Blend._process_blend(img)


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
    def get_homography(img1, img2, bb1, bb2):
        """
        Calculates and filters ORB descriptors
        """

        def _get_orb_features(img, bb1, bb2):
            """
            Returns filtered ORB keypoints and descriptors
            """
            orb = ORB(img)
            kps = orb.get_keypoints()
            filtered_kps = filter_keypoints(bb1, bb2, kps)
            ds = orb.get_descriptors(filtered_kps)

            return filtered_kps, ds

        with ThreadPoolExecutor() as executor:
            t1 = executor.submit(_get_orb_features, img1, bb1, bb2)
            t2 = executor.submit(_get_orb_features, img2, bb1, bb2)

            kps1, ds1 = t1.result()
            kps2, ds2 = t2.result()

        H = find_homography(kps1, ds1, kps2, ds2)

        return H

    @staticmethod
    def _process_blend(img):
        """
        1. color corrects image
        2. extracts face and body bounding boxes

        Returns
         - color corrected image
         - face and
         - body bounds an image with the bounds drawn
        """
        # color correct images
        color_corrected = Blend.color_correction(img)

        # get face and body bounds
        face_bounds, body_bounds = Blend.get_face_body_bounds(color_corrected)

        illustrated_bounds = processing_helpers.draw_box(color_corrected, face_bounds)
        illustrated_bounds = processing_helpers.draw_box(
            illustrated_bounds, body_bounds
        )

        return color_corrected, face_bounds, body_bounds, illustrated_bounds

    def blend(self):
        """
        Performs the FriendBlend algorithm
        """
        p = Pool(2)
        r1, r2 = p.map(_process_blend, [self.img1, self.img2])

        cc1, fb1, bb1, boxed1 = r1
        cc2, fb2, bb2, boxed2 = r2

        # imshow(boxed1)
        # imshow(boxed2)

        # compute homography (uses ORB)
        H = Blend.get_homography(cc1, cc2, bb1, bb2)

        warp_img = cv.warpPerspective(cc1, H, cc1.shape[:2][::-1])
        # imshow(np.hstack([warp_img, cc2]))

        return H


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )
    log = logging.getLogger()

    global_vars.initialize()

    if len(sys.argv) != 3:
        log.warning(
            "Please provide name of the images. (inside `images` directory at repo root)."
            " Using default images as fallback for demonstration"
        )

        img1_path = "../images/f1.png"
        img2_path = "../images/f2.png"
    else:
        img1_path = f"../images/{sys.argv[1]}"
        img2_path = f"../images/{sys.argv[2]}"

    blend = Blend(img1_path, img2_path)

    blend.blend()
