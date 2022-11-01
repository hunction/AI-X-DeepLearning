from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import img_aug as ia

dir_c = './train_img_default(-3)/cafe/'
dir_r = './train_img_default(-3/restaurant/'
categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


model = load_model('ResNet_example.h5')

img_c , label_c = ia.load_img(dir_c)
img_r , label_r = ia.load_img(dir_r)

img_c = np.array(img_c)
img_r = np.array(img_r)
label_c = np.array(label_c)
label_r = np.array(label_r)

img_c = cv2.resize(img_c, (200,200) , cv2.INTER_AREA)
img_r = cv2.resize(img_r, (200,200) , cv2.INTER_AREA)

X = np.vstack(img_c,img_r)
Y = np.hstack(label_c,label_r)

y_hat = model.predict(X)

print(y_hat)