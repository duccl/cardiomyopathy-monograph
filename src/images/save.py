import matplotlib.pyplot as plt
import os
import numpy as np

def write_image(path,image):
    plt.imsave(path,image,cmap='gray')

def save_generated_images(base_path, patient_metadata ,images4d, images_ground_truth4d, ground_truth_suffix = 'gt', image_type ='png'):
    assert images4d.shape == images_ground_truth4d.shape

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for current_time in range(images4d.shape[0]):
        for current_slice in range(images4d.shape[1]):
            random = np.random.randint(0,images4d.shape[0]*images4d.shape[1])
            
            file_path_image = os.path.join(
                base_path,
                f'{patient_metadata.patient_number}_time_{current_time}_slice_{current_time}_{random}.{image_type}'
            )

            file_path_image_ground_truth = os.path.join(
                base_path,
                f'{patient_metadata.patient_number}_time_{current_time}_slice_{current_time}_{random}_{ground_truth_suffix}.{image_type}'
            )

            write_image(file_path_image,images4d[current_time,current_slice])
            write_image(file_path_image_ground_truth,images_ground_truth4d[current_time,current_slice])
