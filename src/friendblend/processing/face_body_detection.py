"""
Face and body detection
"""

import sys

import cv2 as cv

__all__ = ["get_face", "get_body", "get_bounds"]


def get_face(img):
    """
    Returns the face bounds
    Can die in between
    """
    img = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier()

    if not face_cascade.load(
        cv.samples.findFile("./haarcascade_frontalface_default.xml")
    ):
        print("--(!)Error loading face cascade")
        sys.exit(1)

    try:
        face = face_cascade.detectMultiScale(img_gray, minNeighbors=7)[0]
    except IndexError as e:
        print(e)
        print("--(!)No faces detected")
        sys.exit(1)

    return face


def get_body(img, face_bounds):
    """
    Calculates the body bounds using face bounds
    """
    # image height
    hi = img.shape[0]

    # face bounds
    xf, yf, wf, hf = face_bounds

    # x_body_left = x_face_left - wf
    x = xf - wf
    # y_body_top = y_face_top - hf
    y = yf - hf
    w = 3 * wf
    h = hi - y

    return (x, y, w, h)


def get_bounds(img):
    """
    Calculates and returns face and body bounds
    Can die in between
    """
    face_bounds = get_face(img)
    body_bounds = get_body(img, face_bounds)

    return face_bounds, body_bounds
