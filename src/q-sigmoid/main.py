import cv2
import numpy as np
import formula
import displayer

def getSigmoidImage(img_arr):
	img_gray = cv2.cvtColor(np.float32(img_arr/np.max(img_arr)),cv2.COLOR_RGB2GRAY)
	result = np.zeros(img_gray.shape)
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i][j] = formula.q_sigmoid(img_gray[i][j],0.2,0.15,1,0.1)
	return result

img_cmd = cv2.imread('patient001_frame01_1.png')
img_cmd_qsigmoid = getSigmoidImage(img_cmd)

displayer.display_images([img_cmd,img_cmd_qsigmoid], cols = 2, titles = ['Cardiomiopatia Dilatada','Cardiomiopatia Dilatada - q-Sigmoid Aplicado'])
