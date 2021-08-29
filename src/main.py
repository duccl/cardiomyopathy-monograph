from common.download_images import download_patient_data,get_patients_metadata
from incor_masks.read_roi_locations import get_ROI_from_path
from images.load import load_images_4d
from incor_masks.create_mask import create_segmentation
from images.save import save_generated_images
import os
import shutil
from pathlib import Path

def main():
    
    home_path = Path.home()

    roi_locations = get_ROI_from_path(r"..\resources\ROI_locations.txt")
    print('ROI LOCATION OK!')

    patients_metadata,_ = get_patients_metadata()
    print('METADATA OK!')

    for patient_metadata in patients_metadata:
        download_patient_data(patient_metadata)
        print(f'{patient_metadata.patient_number} DOWNLOAD METADATA OK!')

        image4d = load_images_4d(patient_metadata.folder_name)
        segmentation_4d = create_segmentation(patient_metadata.folder_name,image4d,roi_locations[patient_metadata.patient_number])
        print(f'{patient_metadata.patient_number} IMAGES OK!')

        save_generated_images(
            os.path.join(home_path,patient_metadata.folder_name),
            patient_metadata,
            image4d,
            segmentation_4d
        )
        print(f'{patient_metadata.patient_number} IMAGES SAVED OK!')
        
        shutil.rmtree(patient_metadata.folder_name)
        print(f'{patient_metadata.patient_number} FOLDER REMOVED OK!')

if __name__ == '__main__':
    main()