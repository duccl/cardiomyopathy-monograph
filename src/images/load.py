import os
import pydicom
import numpy as np
from pathlib import Path 

from pydicom.filereader import dcmread 

def get_images_paths_by_patient(patient_path:str):
    paths = []
    for root,folders,files in os.walk(patient_path,topdown= False):
        for file in files:
            path = os.path.join(root,file)
            if Path(path).is_file() and  (".dcm" in Path(path).name or "." not in Path(path).name):
                paths.append(path)
    return paths

def load_images_from_paths(paths):
    imgs = {}
    for path in paths:
        dcm =  pydicom.dcmread(path)

        if "TriggerTime" in dcm:
            time = int(dcm["TriggerTime"].value)
        else:
            time = int(dcm["InstanceNumber"].value)

        slice = round(dcm["ImagePositionPatient"].value[-1],2) # slice location

        if slice not in imgs:
            imgs[slice] = {}

        imgs[slice][time] = dcm.pixel_array # image array
    _imgs = imgs
    return imgs

def get_time_size(paths,images):
    first_image = dcmread(paths[0])
    first_key = list(images)[0]
    if "CardiacNumberOfImages" in first_image:
        time_size = first_image["CardiacNumberOfImages"].value

        for k in images:
            assert len(images[k].keys()) == time_size #make sure the time instants are correct

        return time_size
    return len(images[first_key].keys())

def create_4d_image(images, time_size):
    first_time_key = list(images)[0]
    first_slice_time_key = list(images[first_time_key])[0]
    image4d = np.zeros(
        shape=(
            time_size,
            len(images),
            images[first_time_key][first_slice_time_key].shape[0],
            images[first_time_key][first_slice_time_key].shape[1]
        ),
        dtype=np.int16
    )
    for i,slice in enumerate(sorted(images.keys(),reverse=True)):
        for j,time in enumerate(sorted(images[slice].keys())):
            image4d[j,i] = images[slice][time]
    return image4d

def load_images_4d(patient_path: str):
    patient_images_paths = get_images_paths_by_patient(patient_path)
    images = load_images_from_paths(patient_images_paths)
    time_size = get_time_size(patient_images_paths, images)
    return create_4d_image(images,time_size)