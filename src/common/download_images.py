from common.patient_paths import PatientPaths
from pathlib import Path
from typing import List,Tuple
import boto3
import os

def get_prefixes(boto3_objects_response):
    if 'CommonPrefixes' not in boto3_objects_response:
        return []
    return sorted([folder['Prefix'] for folder in boto3_objects_response['CommonPrefixes']],reverse=True)

def download_patient_data(patient_path: PatientPaths, root_path = '.', bucket_name='mri-to-tcc'):
    patient_local_dir = os.path.join(root_path,patient_path.folder_name)
    s3_client = boto3.client('s3')
    if not os.path.exists(patient_local_dir):
        os.makedirs(patient_local_dir)

    mat_file_local_dir = os.path.join(root_path,patient_path.mat_file_path)
    if not os.path.exists(mat_file_local_dir):
        s3_client.download_file(bucket_name,patient_path.mat_file_path,mat_file_local_dir)

    for image in patient_path.images_path:
        if not os.path.exists(Path(image).parent):
            os.makedirs(Path(image).parent)
        s3_client.download_file(bucket_name,image,os.path.join(root_path,image))
    
def get_patients_metadata() -> Tuple[List[PatientPaths],List[PatientPaths]]:
    bucket = 'mri-to-tcc'
    s3 = boto3.client('s3')
    meta_root_folders = s3.list_objects(Bucket='mri-to-tcc',Delimiter='/')
    root_folders = get_prefixes(meta_root_folders)
    patients = []
    faulty_patients = []

    for folder in root_folders:

        if 'Ignore' in folder :
            continue
        
        patients_prefixes = get_prefixes(s3.list_objects_v2(Bucket=bucket,Delimiter='/',Prefix=folder))
        for patient_prefix in patients_prefixes:
            patient_number = Path(patient_prefix).name
            try:
                patient_objects = s3.list_objects_v2(Bucket=bucket,Delimiter='/',Prefix=patient_prefix)
                patient_objects_prefix = next(filter(lambda prefix: '/' in prefix ,get_prefixes(patient_objects)))
                mat_file_prefix = next(filter(lambda prefix: '.mat' in prefix['Key'], patient_objects['Contents']), {'Key':''})['Key']
                images_path = list(
                    map(
                        lambda content: content['Key'],
                        s3.list_objects_v2(Bucket='mri-to-tcc',Delimiter='/',Prefix=patient_objects_prefix)['Contents']
                    )
                )
                patients.append(
                    PatientPaths(patient_prefix,images_path,mat_file_prefix,patient_number,folder)
                )
            except Exception as e:
                print(f'Exception processing patient {patient_prefix}: {e}')
                faulty_patients.append(patient_prefix)
    return patients, faulty_patients