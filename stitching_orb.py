# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:19:59 2019

@author: Dries Ottevaere
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "Images/"
image_start = 800
image_count = 2
image_type = ".jpg"

image_step = 3


images = []
kp = []
des = []
matches = []

orb = cv2.ORB_create()
flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)



def show_keypoints(img,kp):
    
    outImage =	cv2.drawKeypoints(img, kp, None, color=(0,0,255),flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#    print(len(kp))
    plt.imshow(outImage)
#    plt.scatter(kp.pt[0],kp.pt[1],s = 10,c = "red")
    plt.show()

for i in range(0,image_count):
    images.append(plt.imread(image_path+"IMG_0"+str(image_start+i*image_step)+image_type))
    #apply lens filter?
    temp = orb.detectAndCompute(images[i],None)

    kp.append(temp[0])
    des.append(np.float32(temp[1]))
    show_keypoints(images[i],kp[i])


distance = int(images[0].shape[0]/10)

#for i in range(0,1):    
#    temp = []
#    for j,(k1,d1) in enumerate(zip(kp[i],des[i])):  
#        nb_des = []
##        nb_kp = []
#        for l,(k2,d2) in enumerate(zip(kp[i+1],des[i+1])):
#            if(abs(k1.pt[0]-k2.pt[0])<distance and abs(k1.pt[1]-k2.pt[1])<distance):
#                nb_des.append(d2)
##                nb_kp.append(k2)
#        if(nb_des):
#            tmp = bf.match(np.asarray([d1]),nb_des)
#            print(np.asarray([d1]))
#            print(nb_des)
#            temp.append(sorted(tmp, key = lambda x:x.distance))
#            print(temp)
#    matches.append(temp)
    
for i in range(0,1):
    matches.append(flann.knnMatch(des[0], des[1], 2))

#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.7
good_matches = []
for m,n in matches[0]:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)


src = np.float32([kp[0][m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dest = np.float32([kp[1][m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)


plt.imshow(images[0])
plt.scatter(src[:,0,0],src[:,0,1],c ='red')
plt.show()
plt.imshow(images[1])
#plt.scatter(dest[:,0,0],dest[:,0,1],c ='red')
M,mask = cv2.findHomography(dest,src,cv2.RANSAC,5.0)

print(M)

dst = cv2.warpPerspective(images[1],M,(2*images[0].shape[1], 2*images[0].shape[0]))
plt.imshow(dst)
plt.title("Warped Image")
plt.show()
plt.figure()


dst[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]
plt.imshow(dst)
plt.show()





test = np.array([1, 2, 3])
print(test.shape)
