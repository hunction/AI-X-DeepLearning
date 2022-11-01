import img_aug as ia
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob ,os
import random
from sklearn.model_selection import train_test_split

dir_c = './train_img_aug(150)/cafe_aug/'
dir_r = './train_img_aug(150)/restaurant_aug/'
categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

img_c ,label_c = ia.load_img(dir_c)
img_r ,label_r = ia.load_img(dir_r)

img_c = np.array(img_c)
img_r = np.array(img_r)
label_c = np.array(label_c)
label_r = np.array(label_r)


img_c = cv2.resize(img_c,(720,720),cv2.INTER_AREA)
img_r = cv2.resize(img_r,(720,720),cv2.INTER_AREA)

#X = np.vstack([img_c,img_r])
#Y = np.hstack([label_c,label_r])

X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(img_c,label_c)
xy_c = (X_train_c, X_test_c), (Y_train_c, Y_test_c)

X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(img_r,label_r)
xy_r = (X_train_r, X_test_r), (Y_train_r, Y_test_r)


np.save("./c_dataset.npy", xy_c)

np.save("./r_dataset.npy", xy_r)
