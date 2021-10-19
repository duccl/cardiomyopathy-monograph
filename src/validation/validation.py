import numpy as np
from skimage import metrics
from medpy.metric.binary import dc

ef dice_index_medpy(gt,pred):
    argmax_gt = np.argmax(np.copy(gt),axis=-1) 
    argmax_pred = np.argmax(np.copy(pred),axis=-1)
    res = []
    for label in np.unique(argmax_gt):
        label_gt = np.copy(argmax_gt)
        label_pred = np.copy(argmax_pred)
        label_gt[label_gt != label] = 0
        label_pred[label_pred != label] = 0
        label_gt[label_gt == label] = 1
        label_pred[label_pred == label] = 1
        res += [dc(label_gt,label_pred)]
    return np.mean(res)


def hausdorff_index(output,expected_output):
    return hausdorff_distance(output, expected_output)
