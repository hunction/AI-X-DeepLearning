import numpy as np
import matplotlib.pyplot as plt

import cv2

import glob ,os
import random

dir_c = './train_img_default(15)/cafe/'
dir_r = './train_img_default(15)/restaurant/'
categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
dir_c_a = './train_img_default(15)/cafe_aug/'
dir_r_a = './train_img_default(15)/restaurant_aug/'

def load_img(dir):

    X = []
    Y = []

    for index , category in enumerate(categories):
        label = index + 1
        img_dir = dir + category +'/'

        for top, directory , f  in os.walk(img_dir):
            for filename in f:
                img = cv2.imread(img_dir + filename , cv2.IMREAD_COLOR)
                #img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                Y.append(label)

    return X , Y


def fill(img, height, width):
    img = cv2.resize(img, (height, width), cv2.INTER_AREA)
    return img


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    height, width = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (width, height))
    img_ro = fill(img, height, width)

    if angle > 0:
        i = 1
        for i in range(len(img_ro) - 1):
            if (img_ro[:i, 0, 0] == 0).all() and (img_ro[:i + 1, 0, 0] > 0).any():
                start = i
                img_new = img_ro[start:-start, start: -start]
                img_new = fill(img_new, 1440, 1440)
                return img_new

    elif angle < 0:
        i = 1
        for i in range(len(img_ro) - 1):
            if (img_ro[0, :i, 0] == 0).all() and (img_ro[0, :i + 1, 0] > 0).any():
                start = i
                img_new = img_ro[start:-start, start: -start]
                img_new = fill(img_new, 1440 , 1440)
                return img_new

    elif angle == 0:
        return img

def cut(img , pixel):
    pixel = int(random.uniform(-pixel , pixel))

    if pixel > 0:
        img_c = img[pixel:-pixel , :]
        img_c = fill(img_c, 1440 , 1440)
        return img_c

    elif pixel < 0:
        img_c = img[: , -pixel : pixel]
        img_c = fill(img_c , 1440 , 1440)
        return img_c

    elif pixel == 0:
        return img

img_c ,label_c = load_img(dir_c)
img_r ,label_r = load_img(dir_r)

''' img_aug'''
#cafe
for i in range(10):
    z = 1
    for img,label in zip(img_c,label_c):
        img_base = fill(img , 1440 , 1440)
        img_base = rotation(img_base , 30 )
        img_aug = cut(img_base , 100 )
        cv2.imwrite(dir_c_a  + '%d' % label + '/%d_%d.jpg'% (z, i) , img_aug)
        z += 1


#restaurant
for i in range(10):
    z = 1
    for img,label in zip(img_r,label_r):
        img_base = fill(img , 1440 , 1440)
        img_base = rotation(img_base , 30 )
        img_aug = cut( img_base , 100 )
        cv2.imwrite(dir_r_a + '%d' % label + '/%d_%d.jpg'% (z, i) , img_aug)
        z += 1



