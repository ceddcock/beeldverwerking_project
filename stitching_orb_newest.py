# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:19:59 2019

@author: Dries Ottevaere
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_keypoints(img,kp):    
    outImage =	cv2.drawKeypoints(img, kp, None, color=(0,0,255),flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#    print(len(kp))
    plt.imshow(outImage)
#    plt.scatter(kp.pt[0],kp.pt[1],s = 10,c = "red")
    plt.show()
    
    
def read_images(image_path,img_name,image_start,image_stop,blurry_images, image_step = 3,image_type = ".jpg"):
    orb = cv2.ORB_create()
    
    images = []
    used_images = []
    kp = []
    des = []
    i = 0
    current_image = image_start
    while (current_image <= image_stop):
        if(current_image in blurry_images):
            print("image ",current_image," is blurry;ignored")
        else:
            print(image_path+image_name+str(current_image)+image_type)
            images.append(plt.imread(image_path+image_name+str(current_image)+image_type))
            #apply lens filter?
            temp = orb.detectAndCompute(images[-1],None)
        
            kp.append(temp[0])
            des.append(np.float32(temp[1]))
            show_keypoints(images[-1],kp[-1])
            i = i+1
            used_images.append(current_image)
        current_image = current_image+image_step
        
    return images,kp,des,i,used_images

def match_kp(des1,kp1,des2,kp2):
    matches = []

    flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
#    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)    
        
    matches.append(flann.knnMatch(des1, des2, 2))
    
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m,n in matches[0]:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)            
    return good_matches

def perspective_transform(good_matches,kp1,kp2,img1,img2,M,n1,n2):
    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dest = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    
    
    plt.subplot(121)
    plt.imshow(img1)
    plt.title("previous"+str(n1))
    plt.scatter(src[:,0,0],src[:,0,1],c ='red')
    plt.subplot(122)
    plt.imshow(img2)
    plt.title("next"+str(n2))
    plt.scatter(dest[:,0,0],dest[:,0,1],c ='red')
    plt.show()
    #plt.scatter(dest[:,0,0],dest[:,0,1],c ='red')
    M2,mask = cv2.findHomography(dest,src,cv2.RANSAC,5.0)
    
#    M2 = np.matmul(M,M2)

    if((src.shape[0]>=4 or dest.shape[0]>=4) and M2 is not None):    
        dst = cv2.warpPerspective(img2,M2,(2*img1.shape[1], 2*img1.shape[0]))
        plt.imshow(dst)
        plt.title("Warped Image")
        plt.show()
        
#        plt.title("stitched image")
#        dst[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]
#        plt.imshow(dst)
#        plt.show()
    
        return dst,M2
    else:
        print("no homography found")
        return np.zeros((2*img1.shape[1], 2*img1.shape[0])),np.identity(3)

image_path = "Images/"
images = (791,797)
blurry_images = (782,788,790,793,795,805,807,811,813,816,819,821,822,824,827,829,835,837,839,842)
image_start = 781
image_stop = 783#846
#image_count = 10
image_type = ".jpg"
image_step = 2
image_name = "IMG_0"

M = np.identity(3)
                            
images,kp,des,image_count,used_images = read_images(image_path,image_name,image_start,image_stop,blurry_images, image_step,image_type = ".jpg")
#
#for i in range(image_count-1):
#    good_matches = match_kp(des[i],kp[i],des[i+1],kp[i+1])
#
#
#    warped_image,M = perspective_transform(good_matches,kp[i],kp[i+1],images[i],images[i+1],M,used_images[i],used_images[i+1])

# Read the images to be aligned
im1 =  images[0]
im2 =  images[1]
 
# Convert images to grayscale
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 
# Find size of image1
sz = im1.shape
 
# Define the motion model
warp_mode = cv2.MOTION_HOMOGRAPHY
 
# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)
 
# Specify the number of iterations.
number_of_iterations = 5000;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
 
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography 
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
 
# Show final results
cv2.imshow("Image 1", im1)
cv2.imshow("Image 2", im2)
cv2.imshow("Aligned Image 2", im2_aligned)
cv2.waitKey(0)

