from ast import literal_eval

def get_ROI_from_path(roi_locations_path):
    '''
        Summary:
            Loads the file ROI_locations.txt, that contains the corresponding ROI point on the original image by each patient
    '''
    if not roi_locations_path:
        return {}
    with open(roi_locations_path,"r") as file:
        lines = file.read().splitlines() 
        patients = {}
        for line in lines:
            name,ROI = str.split(line,";")
            patients[name] = literal_eval(ROI)
        return patients