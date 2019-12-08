from Registration import Registration
import numpy as np
filepath = "../testbeelden/" # path to images
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from matplotlib import pyplot as plt
from random import sample
from skimage.measure import ransac
from skimage.transform import AffineTransform
######added imports
import math
from skimage.measure import compare_ssim as ssim  
from skimage.measure import compare_mse as mse      
from skimage.measure import compare_nrmse as nrmse  
from skimage.measure import compare_psnr as psnr 
######end added imports
import cv2

def gaussian_weights(window_ext, sigma):
    y, x = np.mgrid[-window_ext:window_ext + 1, -window_ext:window_ext + 1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x ** 2 / sigma ** 2 + y ** 2 / sigma ** 2))
    g /= 2 * np.pi * sigma * sigma
    return g

###################edited Dries############################
def peak_signal_noise_ratio(window_orig,window_warped,weights=1):
#    MSE = mean_square_error(window_orig,window_warped,weights)
#    R = 1 
#    PSNR = 20*math.log10(R/math.sqrt(MSE))
    PSNR = psnr(window_orig,window_warped)
    return PSNR

def mean_square_error(window_orig,window_warped,weights=1):
#    MSE = np.mean(weights*(window_orig - window_warped) ** 2)  
    MSE = mse(window_orig,window_warped)
    return MSE

#weights = boolean
def normalized_root_mean_square_error(window_orig,window_warped,weights=False):
#    MSE = math.sqrt(np.mean(waights*(window_orig - window_warped) ** 2))/np.mean(window_orig - window_warped) 
    NRMSE = nrmse(window_orig,window_warped)
    return NRMSE

def sum_of_square_dist(window_orig,window_warped,weights=1):    
    SSD = np.sum(weights*(window_orig-window_warped)**2) 
    return SSD

#weights = boolean
def structural_similarity_index(window_orig,window_warped,weights=False):
    SSIM = ssim(window_orig,window_warped,gaussian_weights=weights)
    return SSIM
 

def find_matches(coords_orig, coords_warped, im1_coefs, im2_coefs, window_ext=5, match_max_dist=200, sigma=1):
    progress_res = 10
    progress = progress_res
    src, dst = [], []
    weights = gaussian_weights(window_ext,sigma)
    for i, coord in enumerate(coords_orig):
        r, c = np.round(coord).astype(np.intp)
        window_orig = im1_coefs[r - window_ext:r + window_ext + 1,
                      c - window_ext:c + window_ext + 1]

        # compute sum of squared differences to all corners closer than match_max_dist pixels in warped image
        values = []
        for j, (cr, cc) in enumerate(coords_warped):
            try:
                if abs(r - cr) < match_max_dist and abs(c - cc) < match_max_dist:
                    window_warped = im2_coefs[cr - window_ext:cr + window_ext + 1,
                                    cc - window_ext:cc + window_ext + 1]
                    
                    value = sum_of_square_dist(window_orig,window_warped,weights)
                    values.append({'VAL': value, 'corner_idx': j})
            except:
                print("")
        # use corner with minimum val as correspondence (if there are any)
        if values:
            src.append(coord)
            min_idx = np.argmin([ssd['VAL'] for ssd in values])
            dst.append(coords_warped[values[min_idx]['corner_idx']])
        if 100 * i / len(coords_orig) > progress:
            print("\rprogress: {}% of corners done".format(progress), end='')
            progress += progress_res
    print("\rprogress: 100% of corners done", end='')
    return src,dst
####################edited end#####################################

reg = Registration(filepath)
index1 = 0  # index van eerste foto
index2 = 1  # index van derde foto
Harris_th_rel = 0.001
Harris_min_dist = 5
window = 5
#transform, corners, inliers = reg.get_transform(index1, index2, match_max_dist=100) # (mag ook de naam van het bestand zijn of de pixels zelf ipv index)
# transform = transformatiematrix, maar je kan ook de translatie, rotatie en schaal afzonderlijk opvragen met transform.rotation,...
# corners = alle gevonden hoeken in img1 in corners['src'] + beste match in img2 bij overeenkomstige index in corners['dst']
# inliers = aanwezigheidslijst van de gekozen corners (zie foto die grplot wordt)

# dit is om te tonen welke corners gekozen zijn
#

img1 = reg.get_image(index1, gray=False)
img2 = reg.get_image(index2, gray=False)

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

coef1, coef2 = reg.get_swt_coeffs(img1_gray), reg.get_swt_coeffs(img2_gray)
coef1_L = coef1[0]
coef2_L = coef2[0]
LH1,HL1 = coef1[1][0], coef1[1][1]
LH2,HL2 = coef2[1][0], coef2[1][1]
coords_orig_LH = list(corner_peaks(corner_harris(LH1), threshold_rel=Harris_th_rel,min_distance=Harris_min_dist))
coords_orig_HL = list(corner_peaks(corner_harris(HL1), threshold_rel=Harris_th_rel,min_distance=Harris_min_dist))
coords_warped_LH = list(corner_peaks(corner_harris(LH2), threshold_rel=Harris_th_rel,min_distance=Harris_min_dist))
coords_warped_HL = list(corner_peaks(corner_harris(HL2), threshold_rel=Harris_th_rel,min_distance=Harris_min_dist))

coords_orig = coords_orig_HL
for c in coords_orig_LH:
    coords_orig.append(c)
coords_warped = list(coords_warped_HL)
for c in coords_warped_LH:
    coords_warped.append(c)

if len(coords_orig) < 100:
    LH1, HL1 = coef1[2][0], coef1[2][1]
    coords_orig_LH = list(corner_peaks(corner_harris(LH1), threshold_rel=Harris_th_rel, min_distance=Harris_min_dist))
    coords_orig_HL = list(corner_peaks(corner_harris(HL1), threshold_rel=Harris_th_rel, min_distance=Harris_min_dist))
    coords_orig = coords_orig_HL
    for c in coords_orig_LH:
        coords_orig.append(c)
if len(coords_warped) < 100:
    LH2, HL2 = coef2[2][0], coef2[2][1]
    coords_warped_LH = list(corner_peaks(corner_harris(LH2), threshold_rel=Harris_th_rel, min_distance=Harris_min_dist))
    coords_warped_HL = list(corner_peaks(corner_harris(HL2), threshold_rel=Harris_th_rel, min_distance=Harris_min_dist))
    coords_warped = list(coords_warped_HL)
    for c in coords_warped_LH:
        coords_warped.append(c)

# if len(coords_orig) > 2*len(coords_warped):
#     while len(coords_orig) > 2*len(coords_warped):
#         print("image 2 is blurred, blurring image 1")
#         img1_gray = cv2.blur(cv2.blur(img1_gray, (window, window)),(window,window))
#         coef1 = reg.get_swt_coeffs(img1_gray)
#         coef1_L = coef1[0]
#         LH1, HL1 = coef1[2][0], coef1[2][1]
#         coords_orig_LH = list(
#             corner_peaks(corner_harris(LH1), threshold_rel=Harris_th_rel, min_distance=Harris_min_dist))
#         coords_orig_HL = list(
#             corner_peaks(corner_harris(HL1), threshold_rel=Harris_th_rel, min_distance=Harris_min_dist))
#         coords_orig = coords_orig_HL
#         for c in coords_orig_LH:
#             coords_orig.append(c)
# elif len(coords_orig) < 2*len(coords_warped):
#     print("image 1 is blurred, blurring image 2")
#     img2_gray = cv2.blur(cv2.blur(img2_gray, (window, window)), (window, window))
#     coef2 = reg.get_swt_coeffs(img2_gray)
#     coef2_L = coef2[0]


print(len(coords_orig),len(coords_warped))
src, dst = find_matches(coords_orig, coords_warped, coef1_L, coef2_L,match_max_dist=250,window_ext=5)

src = np.asarray(src)
dst = np.asarray(dst)

src*=2
dst*=2

transform = cv2.findHomography(src,dst,cv2.RANSAC)
print(transform)
print(transform[1])
print(transform[1]==1)
reg.show_matches(img1,img2,src,dst,transform[1]==1)

# model_robust, inliers = ransac((src, dst), PerspectiveTransform, min_samples=5,
#                                        residual_threshold=5, max_trials=100)
# while(sum(inliers)<4):
#     model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=5,
#                                    residual_threshold=5, max_trials=100)

#reg.show_matches(img1,img2,src,dst,inliers)

# print(f"Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), "
#       f"Translation: ({model_robust.translation[0]:.4f}, "
#       f"{model_robust.translation[1]:.4f}), "
#       f"Rotation: {model_robust.rotation:.4f}")