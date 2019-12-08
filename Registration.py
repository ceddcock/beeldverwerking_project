import numpy as np
import cv2
from matplotlib import pyplot as plt
import pywt
from os import listdir


from skimage import data
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

class Registration:

    def __init__(self, filepath):
    # filepath is path with images to be registrated
        self.filepath = filepath
        self.pictures_list = listdir(filepath)

    def get_image(self, data_or_name_or_index, gray=False):
    # returns the image as np.ndarray
    # im = index or name of picture in filepath

        if type(data_or_name_or_index) is int:
            index = data_or_name_or_index
            if gray:
                img = cv2.imread(self.filepath + self.pictures_list[index], cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(self.filepath + self.pictures_list[index])
            return img
        elif type(data_or_name_or_index) is str:
            name = data_or_name_or_index
            if gray:
                img = cv2.imread(self.filepath + name, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(self.filepath + name)
            return img
        elif type(data_or_name_or_index) is np.ndarray:
            return data_or_name_or_index

    def get_transform(self, im1, im2, window_ext=5, sigma=3, wavelet='db1', downsample=True, Harris_th_rel=0.001, Harris_min_dist=5, match_max_dist=300):
    # gives image transform from im1 to im2
    # im1, im2 can be an index, filename or np.ndarray containing the pixels (must be gray scale)
    # window_ext = window size * 0.5 - 1, the window size used to match the corners with MSE
    # sigma is variance of gaussian window over MSE used to give central pixels a higher weight
    # wavelet used for corner matching and downsampling (matching not yet implemented)
    # downsample needs to be true if the pictures are very large

        # getting image wavelet coefficients
        print("\r[getting image wavelet coefficients]")
        print("\rprogress: 0/2 images done", end='')
        im1_coefs = self.get_swt_coeffs(im1, wavelet, downsample)[0]
        print("\rprogress: 1/2 images done", end='')
        im2_coefs = self.get_swt_coeffs(im2, wavelet, downsample)[0]
        print("\rprogress: 2/2 images done", end='')

        print("\r[searching corners in images]")
        print("\rprogress: 0/2 images done", end='')
        # extract corners using Harris' corner measure
        coords_orig = corner_peaks(corner_harris(im1_coefs), threshold_rel=Harris_th_rel,
                                   min_distance=Harris_min_dist)

        print("\rprogress: 1/2 images done", end='')
        coords_warped = corner_peaks(corner_harris(im2_coefs),
                                     threshold_rel=Harris_th_rel, min_distance=Harris_min_dist)

        print("\rprogress: 2/2 images done", end='')
        print("\rfound {} corners in image 1 and {} coreners in image 2".format(len(coords_orig), len(coords_warped)))

        # weight pixels depending on distance to center pixel
        weights = self.gaussian_weights(window_ext, sigma)

        # find correspondences using simple weighted sum of squared differences
        print("\r[finding best matching corner in image 2 for every corner of image 1]")
        src = []
        dst = []
        print("\rprogress: 0% of corners done", end='')
        progress_res = 10
        progress = progress_res
        for i, coord in enumerate(coords_orig):
            r, c = np.round(coord).astype(np.intp)
            window_orig = im1_coefs[r - window_ext:r + window_ext + 1,
                          c - window_ext:c + window_ext + 1]

            # compute sum of squared differences to all corners closer than match_max_dist pixels in warped image
            SSDs = []
            for j, (cr, cc) in enumerate(coords_warped):
                if abs(r - cr) < match_max_dist and abs(c - cc) < match_max_dist:
                    window_warped = im2_coefs[cr - window_ext:cr + window_ext + 1,
                                    cc - window_ext:cc + window_ext + 1]
                    SSD = np.sum(weights * (window_orig - window_warped) ** 2)
                    SSDs.append({'SSD': SSD, 'corner_idx': j})

            # use corner with minimum SSD as correspondence (if there are any)
            if SSDs:
                src.append(coord)
                min_idx = np.argmin([ssd['SSD'] for ssd in SSDs])
                dst.append(coords_warped[SSDs[min_idx]['corner_idx']])
            if 100 * i / len(coords_orig) > progress:
                print("\rprogress: {}% of corners done".format(progress), end='')
                progress += progress_res

        print("\rMatching done")
        src = np.array(src)
        dst = np.array(dst)

        # double all coordinates if subsampled
        if downsample:
            src *= 2
            dst *= 2

        # robustly estimate affine transform model with RANSAC
        print("\r[Estimating affine transform model with Ransac]")
        model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=4,
                                       residual_threshold=2, max_trials=100)
        outliers = inliers == False
        matched_corners = {"src": src, "dst": dst}
        print("\rEstimation done")
        return model_robust, matched_corners, inliers

    def show_matches(self, im1, im2, src, dst, inliers):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        inlier_idxs = np.nonzero(inliers)[0]
        plot_matches(ax, im1, im2, src, dst,
                     np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b',only_matches=True)
        ax.axis('off')
        ax.set_title('Correct correspondences')

        plt.show()

    def show_swt(self, data_or_name_or_index, l=None):
        # data_or_name_or_index can be wavelet coefficients, name of the file or index of file in folder
        # l is decomposition level
        # plots the coefficients of level l

        if type(data_or_name_or_index) is list:
            coefs = data_or_name_or_index
        else:
            name_or_index = data_or_name_or_index
            coefs = self.get_swt_coeffs(name_or_index)
        N = len(coefs) - 1
        LL, (LH, HL, HH) = coefs[0], coefs[N - l]
        titles = ['LL max level ({})'.format(N - 1), 'LH level {}'.format(l), 'HL level {}'.format(l),
                  'HH level {}'.format(l)]
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap='gray')
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
        plt.show()

    # picture name or index
    def get_swt_coeffs(self, img, wavelet='db1', downsample="True"):

        p = self.get_image(img, gray=True)

        # downsampling
        if downsample:
            p = pywt.dwt2(p, wavelet)[0]

        # calculate maximum decomposition level
        l_max = pywt.swt_max_level(min(np.shape(p)))
        self.max_level = l_max

        # Multilevel 2D stationary wavelet transform with wavelet filters.
        coefs = pywt.swt2(p, wavelet, l_max, start_level=0, trim_approx=True, norm=True)

        # converting from float64 to uint8 to save memory
        coefs[0] = coefs[0].astype(np.uint8)
        for level in range(1, l_max):
            coefs[level] = list(coefs[level])
            for band in range(3):
                picture = np.asarray(coefs[level][band])
                picture -= np.min(picture)
                picture *= 255 / np.max(picture)
                picture = picture.astype(np.uint8)
                coefs[level][band] = picture
        return coefs

    def gaussian_weights(self, window_ext, sigma=1):
        y, x = np.mgrid[-window_ext:window_ext + 1, -window_ext:window_ext + 1]
        g = np.zeros(y.shape, dtype=np.double)
        g[:] = np.exp(-0.5 * (x ** 2 / sigma ** 2 + y ** 2 / sigma ** 2))
        g /= 2 * np.pi * sigma * sigma
        return g








