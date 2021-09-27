from scripts.main import DataAugmentation
import matplotlib.pyplot as plt
from pathlib import Path
import random

ROOT_HIPERTROFICOS = r'C:\Users\duccl\INCOR\Hipertrófico'
ROOT_NORMAL = r'C:\Users\duccl\INCOR\Normal'

QUANTITY_PENDING_HIPERTROFICOS = 417
QUANTITY_PENDING_NORMAL = 143

def get_samples(root_path, samples):
    _samples = {}
    for path in Path(root_path).rglob('*.png'):
        path_name = path.name
        current_patient_frame = path_name.split('.')[0].split('_gt')[0]

        if current_patient_frame not in _samples:
            _samples[current_patient_frame] = []
        
        _samples[current_patient_frame].append(str(path))
    return random.sample( list(_samples.keys()), samples), _samples

def save_used_paths(paths):
    with open('trace_data_augmentation','w') as file:
        for path in paths:
            file.write(str(path) + '\n')


def apply_data_augmentation():
    sample_keys = []
    hipertroficos_keys, hipertroficos_paths = get_samples(ROOT_HIPERTROFICOS,QUANTITY_PENDING_HIPERTROFICOS)
    normal_keys, normal_paths = get_samples(ROOT_NORMAL,QUANTITY_PENDING_NORMAL)
    sample_keys.extend(hipertroficos_keys)
    sample_keys.extend(normal_keys)
    all_paths = {**hipertroficos_paths, **normal_paths}
    used_paths = []

    for key in sample_keys:
        image = plt.imread(all_paths[key][0])
        gt_image = plt.imread(all_paths[key][1])
        image_augmented = DataAugmentation(image).move().rotate().apply()
        gt_image_augmented = DataAugmentation(gt_image).move().rotate().apply()
        image_augmented_path = f"{all_paths[key][0].split('.png')[0]}_data_augmentation.png"
        gt_image_augmented_path = f"{all_paths[key][1].split('.png')[0]}_data_augmentation.png"
        plt.imsave(image_augmented_path,image_augmented)
        plt.imsave(gt_image_augmented_path,gt_image_augmented)
        used_paths.append(all_paths[key])

    save_used_paths(used_paths)

if __name__ == '__main__':
    apply_data_augmentation()