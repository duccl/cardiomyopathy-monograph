class PatientPaths:
    def __init__(self,folder_name,images_path,mat_file_path,patient_number, class_type):
        self.folder_name = folder_name
        self.images_path = images_path
        self.mat_file_path = mat_file_path
        self.patient_number = patient_number
        self.class_type = class_type if '/' not in class_type else class_type.split('/')[0]