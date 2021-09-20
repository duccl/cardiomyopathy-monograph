import tensorflow.keras.backend as K
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff

def dice_coef(y_true, y_pred,smooth=1e-7):
    print(f'y_true min = {tf.reduce_min(y_true)}, y_pred min = {tf.reduce_min(y_pred)}')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def multiclass_hausdorff(y_true,y_pred, labels = 3):
    res = 0
    for label in range(labels):
        res += directed_hausdorff(y_true[:,:,label],y_pred[:,:,label])[0]
    return res