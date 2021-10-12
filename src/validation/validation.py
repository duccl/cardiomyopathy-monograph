import numpy as np
from skimage import metrics

def dice_coef(y_true, y_pred,smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_index(y_true, y_pred, numLabels = 3):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index])
    return dice/numLabels


def hausdorff_index(output,expected_output):
    return hausdorff_distance(output, expected_output)
