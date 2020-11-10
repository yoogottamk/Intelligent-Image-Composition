"""
Contains code for color correction of the image.
CLAHE - Contrast Limited Adaptive Histogram Equalization
"""

import cv2 as cv
import numpy as np

__all__ = ["rgb2lab", "clahe", "apply_clahe"]


def rgb2lab(img):
    """
    Converts an RGB image to L*a*b* space
    Implements formulas described at:
    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    """

    def f(X):
        """
        Helper function for RGB to Lab conversion
        Detailed description in the link above
        """

        X = X.copy()

        mask1 = X > 0.008856
        mask2 = X <= 0.008856

        X[mask1] = X[mask1] ** (1 / 3)
        X[mask2] = (7.787 * X[mask2]) + (16 / 116)

        return X

    # converts image to be in [0,1] range
    img = img.astype("float64")
    img /= 255.0

    # accounts for sRGB conversion
    img = img ** 2.2

    # converts to XYZ first
    X = (
        (img[:, :, 0] * 0.412453)
        + (img[:, :, 1] * 0.357580)
        + (img[:, :, 2] * 0.180423)
    )
    Y = (
        (img[:, :, 0] * 0.212671)
        + (img[:, :, 1] * 0.715160)
        + (img[:, :, 2] * 0.072169)
    )
    Z = (
        (img[:, :, 0] * 0.019334)
        + (img[:, :, 1] * 0.119193)
        + (img[:, :, 2] * 0.950227)
    )

    # initializing constants
    Xn = 0.950456
    Zn = 1.088754
    delta = 0

    # converting from XYZ to Lab as described by the formulas
    X /= Xn
    Z /= Zn

    mask1 = Y > 0.008856
    mask2 = Y <= 0.008856

    L = Y.copy()
    L[mask1] = 116 * (L[mask1] ** (1 / 3)) - 16
    L[mask2] = 903.3 * L[mask2]

    # f(X) defined above
    a = 500 * (f(X) - f(Y)) + delta
    b = 200 * (f(Y) - f(Z)) + delta

    L = L * (255 / 100)
    a = a + 128
    b = b + 128

    return np.round(np.dstack([L, a, b])).astype("uint8")


def _interpolate(c_sub_bin, map_ul, map_ur, map_dl, map_dr, x_frame_size, y_frame_size):
    """
    Interpolates the values of the pixels in patches around the current pixel patch and
    finds the value for the current pixel
    """

    # initializing output
    c_sub_image = np.zeros(c_sub_bin.shape)

    # implementing the formula
    for i in range(x_frame_size):
        i_inv = x_frame_size - i
        for j in range(y_frame_size):
            j_inv = y_frame_size - j

            val = c_sub_bin[i, j].astype(int)
            c_sub_image[i, j] = i_inv * (j_inv * map_ul[val] + j * map_ur[val]) + i * (
                j_inv * map_dl[val] + j * map_dr[val]
            )

    return (c_sub_image / (x_frame_size * y_frame_size)).astype("int32")


def generate_histograms(
    bins, n_bins, grid_x_divs, grid_y_divs, x_frame_size, y_frame_size
):
    """
    Generates the histogram required for each patch
    """
    hist = np.zeros((grid_x_divs, grid_y_divs, n_bins))

    for i in range(grid_x_divs):
        for j in range(grid_y_divs):
            # gets the current frame (it's binned values)
            c_bin = bins[
                i * x_frame_size : (i + 1) * x_frame_size,
                j * y_frame_size : (j + 1) * y_frame_size,
            ]
            c_bin = c_bin.astype("int32")

            # updates the current frame's histogram
            for p in range(x_frame_size):
                for q in range(y_frame_size):
                    hist[i, j, c_bin[p, q]] += 1

    return hist


def clip_histogram(hist, clip_limit, n_bins, grid_x_divs, grid_y_divs):
    """
    Clips the histograms using the limit set
    """
    for i in range(grid_x_divs):
        for j in range(grid_y_divs):
            total_overflow = 0

            for c_bin in range(n_bins):
                overflow = hist[i, j, c_bin] - clip_limit
                if overflow > 0:
                    total_overflow += overflow

            # to redistribute intensities across each bin
            inc_bin_val = total_overflow / n_bins

            # new upper limit for each bin based on how the redistribution has to be done
            upper = clip_limit - inc_bin_val

            # redistributing
            for c_bin in range(n_bins):
                if hist[i, j, c_bin] > clip_limit:
                    hist[i, j, c_bin] = clip_limit
                else:
                    if hist[i, j, c_bin] > upper:
                        total_overflow -= hist[i, j, c_bin] - upper
                        hist[i, j, c_bin] = clip_limit
                    else:
                        hist[i, j, c_bin] += inc_bin_val
                        total_overflow -= inc_bin_val

            # if there is still some overflow, then uniformly divide the overflow into each bin
            if total_overflow > 0:
                div_uniform = max(1, 1 + total_overflow // n_bins)

                for c_bin in range(n_bins):
                    hist[i, j, c_bin] += div_uniform
                    total_overflow -= div_uniform
                    if total_overflow <= 0:
                        break

    return hist


def create_eq_mappings(hist, n_bins, frame_size, grid_x_divs, grid_y_divs):
    """
    Creates the histogram equalization mapping function for each patch
    """
    max_val = 255
    min_val = 0

    maps = np.zeros((grid_x_divs, grid_y_divs, n_bins))
    scale = (max_val - min_val) / float(frame_size)

    for i in range(grid_x_divs):
        for j in range(grid_y_divs):
            c_sum = 0
            for c_bin in range(n_bins):
                c_sum += hist[i, j, c_bin]
                maps[i, j, c_bin] = np.clip((c_sum * scale) + min_val, min_val, max_val)

    return maps


def interpolate(res, maps, bins, grid_x_divs, grid_y_divs, x_frame_size, y_frame_size):
    """
    Wrapper function for the above _interpolate function
    Performs the interpolation on pixels of each patch
    """

    x_offset = 0
    for i in range(grid_x_divs):
        x_up = max(0, i - 1)
        x_down = min(i, grid_x_divs - 1)
        y_offset = 0
        for j in range(grid_y_divs):
            y_left = max(0, j - 1)
            y_right = min(j, grid_y_divs - 1)

            # gets the equalization maps of neighboring patches
            map_ul = maps[x_up, y_left]
            map_ur = maps[x_up, y_right]
            map_dl = maps[x_down, y_left]
            map_dr = maps[x_down, y_right]

            # gets the current bin
            c_sub_bin = bins[
                x_offset : x_offset + x_frame_size, y_offset : y_offset + y_frame_size
            ]

            # gets the interpolated values for current patch
            c_sub_image = _interpolate(
                c_sub_bin, map_ul, map_ur, map_dl, map_dr, x_frame_size, y_frame_size
            )

            # adds current patch interpolated values to the final result
            res[
                x_offset : x_offset + x_frame_size, y_offset : y_offset + y_frame_size
            ] = c_sub_image
            y_offset += y_frame_size
        x_offset += x_frame_size

    return res


def clahe(img, clip_limit=3.0, n_bins=256, grid=(7, 7)):
    """
    - Divides the image into frames/cells using the grid values passed to the function
    - Creates a separate histogram for each cell/frame after binning the values
    - The histogram equalization mapping for each cell/frame is also generated
    - The contrast of the image is limited by clipping bin values with 'clip_limit'
    - Adaptive histogram equalization
    """
    h, w = img.shape
    grid_x_divs, grid_y_divs = list(map(int, grid))

    # find amount of padding needed and apply
    padding_x = grid_x_divs - int(h % grid_x_divs)
    padding_y = grid_y_divs - int(w % grid_y_divs)

    if padding_x != 0:
        img = np.append(img, np.zeros((padding_x, img.shape[1])).astype(int), axis=0)
    if padding_y != 0:
        img = np.append(img, np.zeros((img.shape[0], padding_y)).astype(int), axis=1)

    # initialize result
    res = np.zeros(img.shape)

    # get size of each cell/frame in the grid
    x_frame_size = int(img.shape[0] / grid_x_divs)
    y_frame_size = int(img.shape[1] / grid_y_divs)
    frame_size = x_frame_size * y_frame_size

    # set clip limit
    clip_limit = max(5, clip_limit * x_frame_size * y_frame_size / n_bins)

    # generate Look Up Table and put pixel intensities into their bins
    bin_size = 256.0 / n_bins
    LUT = (np.arange(256) / bin_size).astype("int32")
    bins = LUT[img]

    # creating separate histograms for each frame/cell
    hist = generate_histograms(
        bins, n_bins, grid_x_divs, grid_y_divs, x_frame_size, y_frame_size
    )

    # clipping the histogram to limit contrast
    hist = clip_histogram(hist, clip_limit, n_bins, grid_x_divs, grid_y_divs)

    # creating the equalization mapping
    maps = create_eq_mappings(hist, n_bins, frame_size, grid_x_divs, grid_y_divs)

    # Interpolating the values from surrounding frames
    res = interpolate(
        res, maps, bins, grid_x_divs, grid_y_divs, x_frame_size, y_frame_size
    )

    # return result without padding
    return res[:h, :w].astype("uint8")


def apply_clahe(img, clip_limit=3.0, n_bins=256, grid=(7, 7)):
    """
    Parameters:
     - The input image in BGR format
     - The clip limit for contrast limiting
     - Number of bins to divide the image into
     - Grid size to be used while performing adaptive histogram equalization
    """
    img = img[:, :, ::-1].copy()

    # conversion to Lab space
    img_lab = rgb2lab(img)

    # performing CLAHE on L channel and replacing original L channel with it
    clahe_L = clahe(img_lab[:, :, 0], clip_limit, n_bins, grid)
    img_clahe = np.dstack([clahe_L, img_lab[:, :, 1], img_lab[:, :, 2]])

    return cv.cvtColor(img_clahe, cv.COLOR_LAB2BGR)
