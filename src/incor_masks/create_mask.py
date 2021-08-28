import os
import numpy as np
from scipy import io
from scipy.ndimage.morphology import binary_fill_holes
from skimage.draw import line
import math

def get_mat_file(patient_path_root):
    for root,_,files in os.walk(patient_path_root,topdown=False):
        for file in files:
            path = os.path.join(root,file)
            if '.mat' in file:
                return io.loadmat(path)

def create_ground_truth_mask4d(shape, x, y):
    '''
        Summary:
            Create 4d binary mask using linear interpolation (line) beetween x[i],y[i] and x[i+1],y[i+1] and fills it with binary rules
        Params:
            shape: np.array -> 4d array shape 
            x: np.array -> x coordinates of ROI as time, slice, x_coord
            y: np.array -> y coordinates of ROI as time, slice, y_coord
        Returns:
            4d np.array of time,slice and ground truth binary mask
    '''

    assert len(shape) == 4, f"Shape must be of time, slice, x, y shape! Received {len(shape)} instead"
    assert len(x.shape) == 3 and len(y.shape) == 3, f"x and y must be of time, slice, ROI coord! x.shape = {x.shape}, y.shape = {y.shape}"
    assert x.shape[2] == y.shape[2], f"x and y must have the same quantity of coordinates! x.shape = {x.shape[2]}, y.shape = {y.shape[2]}"

    ground_truth4d = np.zeros(shape)
    for current_time in range(shape[0]):
        for current_slice in range(shape[1]):

            if math.isnan(x[current_time,current_slice,0]):
                continue

            current_image_x = x[current_time,current_slice]
            current_image_y = y[current_time,current_slice]

            # em python, -1 pega o ultimo item do array (caso len(array) > 0)
            # aqui está rodando começando do último até o penúltimo basicamente
            for i in range(-1, x.shape[2]-1):
                x1, y1 = np.array(
                    [
                        min(current_image_x[i],shape[2]-1), # o min é usado pra evitar que valor saia da imagem
                        min(current_image_y[i],shape[3]-1) # o min é usado pra evitar que valor saia da imagem
                    ]
                    ,dtype=np.int32
                )
                x2, y2 = np.array(
                    [
                        min(current_image_x[i+1],shape[2]-1), # o min é usado pra evitar que valor saia da imagem
                        min(current_image_y[i+1],shape[3]-1) # o min é usado pra evitar que valor saia da imagem
                    ]
                    ,dtype=np.int32
                )
                points_that_belongs_to_roi = line(x1,y1,x2,y2)
                current_ground_truth = ground_truth4d[current_time,current_slice]
                current_ground_truth[points_that_belongs_to_roi] = 1            
            ground_truth4d[current_time,current_slice] = binary_fill_holes(ground_truth4d[current_time,current_slice])

    return ground_truth4d

def create_segmentation(patient_path_root, image4d, roi_position):

    assert roi_position, "Damn, you will need to run the brute force method"

    segmentation4d = np.zeros(shape = image4d.shape, dtype = np.dtype(np.uint8) )
    mat_file = get_mat_file(patient_path_root)
    data = mat_file["setstruct"][0][0]
    roi_image = np.transpose(data[0], [2,3,0,1])
    assert roi_image.shape[0] == image4d.shape[0] and roi_image.shape[1] == image4d.shape[1], f"Your time and slice do not match the ROI image! roi_image.shape = {roi_image.shape}, image4d.shape = {image4d.shape}"
    coord = []
    for i in range(len(data)):
        if len(data[i].shape)==3 and data[i].shape[1]==roi_image.shape[0] and data[i].shape[2]==roi_image.shape[1]:
            coord.append(np.transpose(data[i],axes=[1,2,0]))
    assert len(coord) == 4
    x_en,y_en,x_ep,y_ep = coord

    x_en += roi_position[0]
    y_en += roi_position[1]
    x_ep += roi_position[0]
    y_ep += roi_position[1]
    ground_truth_en = create_ground_truth_mask4d(image4d.shape,x_en,y_en)
    ground_truth_ep = create_ground_truth_mask4d(image4d.shape,x_ep,y_ep)

    segmentation4d[ground_truth_en==1] = 1
    segmentation4d[np.logical_and(ground_truth_ep==1,ground_truth_en==0)] = 2

    return segmentation4d