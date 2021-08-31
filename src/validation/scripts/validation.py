import numpy as np
from skimage import metrics

def dice_index(output,expected_output,k = 1):
    return np.sum(output[expected_output==k])*2.0 / (np.sum(output) + np.sum(expected_output))

def hausdorff_index(output,expected_output):
    return hausdorff_distance(output, expected_output)
