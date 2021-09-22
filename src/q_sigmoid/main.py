from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import formula
from PIL import ImageChops, Image, ImageStat
from sklearn.metrics import jaccard_score
import tensorflow as tf
SM_FRAMEWORK=tf.keras
import json
from skimage.transform import resize
from ast import literal_eval as make_tuple
import os
import numpy as np
from pathlib import Path
import nibabel as nib
from tensorflow.keras import backend as K
import tensorflow as tf
import nibabel as nib
import cv2
import random
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import *
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm
import pandas as pd
from segmentation_models.base.functional import iou_score

from loader import load_acdc

def getSigmoidImage(img_arr):
	# img_gray = cv2.cvtColor(np.float32(img_arr/np.max(img_arr)),cv2.COLOR_RGB2GRAY)
	result = np.zeros(img_arr.shape)
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i][j] = formula.q_sigmoid(I = img_arr[i][j],beta=2,alfa=0.51,q=.1,lamb=1)
	return result

def getVectorSigmoidImage(img_arr):
	# img_gray = cv2.cvtColor(np.float32(img_arr/np.max(img_arr)),cv2.COLOR_RGB2GRAY)
	result = np.zeros(img_arr.shape)
	result = formula.q_sigmoid_vector(image=img_arr,beta=2,alfa=0.51,q=.1,lamb=0.41)
	return result

img_cmd = load_acdc(r"D:\Documents\TCC\ac dc\training\training\patient001\patient001_frame01.nii.gz")[0]
# img_cmd = plt.imread('0_true_2.png')
# print(img_cmd.shape)
# img_cmd = np.argmax(img_cmd,axis=-1)
print(datetime.now())
img_cmd_qsigmoid = getSigmoidImage(img_cmd)
print(datetime.now())
# img_cmd = plt.imread('0_true_2.png')
# img_cmd = np.argmax(img_cmd,axis=-1)
print(datetime.now())
img_cmd_qsigmoid_vector = getVectorSigmoidImage(img_cmd)
print(datetime.now())

# cv2.imshow('./Cardiomiopatia Dilatada.png',img_cmd.astype(np.float32))
# cv2.imshow('Cardiomiopatia Dilatada - q-Sigmoid Aplicado',img_cmd_qsigmoid.astype(np.float32))
# cv2.imshow('Cardiomiopatia Dilatada - q-Sigmoid Aplicado Vetor',img_cmd_qsigmoid_vector.astype(np.float32))


# def jaccard_binary(x,y):
#     """A function for finding the similarity between two binary vectors"""
#     intersection = np.logical_and(x, y)
#     union = np.logical_or(x, y)
#     similarity = intersection.sum() / float(union.sum())
#     return similarity


# # a=iou_score(img_cmd_qsigmoid, img_cmd_qsigmoid_vector,backend=tf.keras)
# # print(a)
# js = jaccard_binary(img_cmd_qsigmoid, img_cmd_qsigmoid_vector)
# print(js)

# # im1 = Image.fromarray(img_cmd_qsigmoid)
# # im2 = Image.fromarray(img_cmd_qsigmoid_vector)
# # diff = ImageChops.difference(im1, im2)
# # stat = ImageStat.Stat(diff)
# # diff_ratio = sum(stat.mean) / (len(stat.mean) * 255)
# # print(diff_ratio)

# print(str(img_cmd_qsigmoid),str(img_cmd_qsigmoid_vector))
# print(str(img_cmd_qsigmoid)==str(img_cmd_qsigmoid_vector))

# # displayer.display_images(
# # 	[img_cmd,img_cmd_qsigmoid],
# # 	cols = 3,
# # 	titles = ['Cardiomiopatia Dilatada','Cardiomiopatia Dilatada - q-Sigmoid Aplicado']
# # )

# cv2.waitKey(0)