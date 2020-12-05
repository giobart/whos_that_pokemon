import os
from torch.utils.data import Dataset 
from tools.dataset_tools import get_dataset_filename_map



class LfwImagesDataset(Dataset):


image_map = get_dataset_filename_map()
    """ Face dataset. """
        def __init__(self, data_root):
            self.data_root = data_root
            image_map = get_dataset_filename_map()
            labels = []
        self.encode_classes()

# Encode to have indices to iterate over

 def encode_classes(self):
     loops = 0
     while loops < 1:
    for key in image_map:
        self.class_to_idx = dict()
        for filename in self.image_map:
            split_path = filename.split(os.sep)
            label = split_path[-2]
            self.class_to_idx[label] = self.class_to_idx.get(label, len(self.class_to_idx))


get id

 def __getitem__(self, idx):

          
          image = Image.open(self.image_filenames[idx])
          image2 = Image.open(self.image_filenames[random])
        split_path = self.image_filenames[idx].split(os.sep)
        label = split_path[-2]        
        return image, image, labels






def __getitem__(self, idx, bool)
    image = Image.open(self.image_filenames[idx])
    image2 = Image.open(self.image_filenames[idx])
  #if true return 2 random images of this person and the label with path 

        if bool = true
        
        # append 2 labels to the list of labels
        while 
            labels.append(image_map[keys])
        return image1, image2, labels

    # false and return 2 images of different persons
        else


        return image1, image2, labels

