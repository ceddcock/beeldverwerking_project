from Registration import Registration
import numpy as np
filepath = "../testbeelden/" # path to images
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from matplotlib import pyplot as plt
from random import sample
from skimage.measure import ransac
from skimage.transform import AffineTransform

import cv2

reg = Registration(filepath)
index1 = 0  # index van eerste foto
index2 = 1  # index van derde foto

src = np.asarray([[1082, 541], [716, 1883], [1222, 3047], [1837, 2881], [2454, 623], [2784, 1197], [2501, 2891], [2453, 622]])
dst = np.asarray([[1114, 732], [740, 2046], [1295, 3218], [1940, 3044], [2488, 757], [2861, 1311], [2619,3052], [2490, 755]])

img1 = reg.get_image(index1, gray=False)
img2 = reg.get_image(index2, gray=False)
row,col,chan = img1.shape

transform = cv2.findHomography(src, dst,cv2.RANSAC)
reg.show_matches(img1,img2,src,dst,transform[1]==1)



img2_warped = cv2.warpPerspective(img2,np.linalg.inv(transform[0]),(col*2,row*2))
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if sum(img2_warped[i][j][:]) == 0:
            img2_warped[i][j][:] = img1[i][j][:]
#empty = sum(img2_warped[i%img1.shape[0]][i/img1.shape[1]][:])
plt.imshow(img2_warped)
plt.show()

# row,col,chan = img1.shape
# img1_warped = cv2.warpAffine(img1,model,(col,row))
# plt.imshow(img1)
# plt.plot()
# plt.imshow(img1_warped)
# plt.plot()
# plt.imshow(img2)
# plt.plot()
