from os import path
import requests
import zipfile
from config import *

TARGET_FOLDER = join(".",DATASET_FOLDER) 
FOLDER_LIST = []

try: 
    FOLDER_LIST=glob.glob(DATASET_FOLDER_IMG)
except:
    print("Unable to load Folder List, try again after downloading the dataset")
    response = input("Download it now? (y,n)")
    if "y" in response:
        dataset_download()

def dataset_download():
    """
        Function used to download the datased with all the info provided into the config file
    """
    r = requests.get(DATASET_URL, allow_redirects=True)
    target_filepath = join(".",DATASET_ZIP_NAME)
    target_folder = join(".",DATASET_MAIN_FOLDER_NAME)   
    open(TARGET_FOLDER, 'wb').write(r.content)
    zipfile = zipfile.ZipFile(target_filepath, "r")
    zipfile.extractall(path=target_folder)
    zipfile.close()
    FOLDER_LIST=glob.glob(DATASET_FOLDER_IMG)

def get_dataset_filename_map(min_val=2):
    """
        Function used to return a map key-value
        Key: Name of the person
        Value: Image path list 
        This function mu
        
        Arguments: 
        min_val= minim amount of element that the Value list must contain. Default: 2
    """
    result = {}
    for folder_path in FOLDER_LIST:
        person_name=os.path.basename(os.path.normpath(folder_path))
        image_list=glob.glob(join(folder_path, "*"+DATASET_IMAGE_EXTENSION))
        if len(image_list)>=min_val:
            result[person_name]=image_list
            
    return result
        