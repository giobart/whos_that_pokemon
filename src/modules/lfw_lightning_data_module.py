import pytorch_lightning as pl
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from random import seed
from PIL import Image
from random import randint
import numpy as np
from torch.utils.data import random_split
from src.tools.image_preprocess import FaceAlignTransform


class LFW_DataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size=32, splitting_points=(0.11, 0.11), num_workers=4, shuffle=False, manual_split=False,
                 valid_dataset=None, test_dataset=None, input_size=128):
        """
        Args:
            dataset: LfwImagesDataset(), if manual_split==True than this is the LfwImagesPairsDataset train set
            batch_size: default value: 32
            splitting_points:   splitting point % for train, test and validation.
                                default (0.6,0.85) -> 60% train, 25% validation, 15% test
            valid_dataset: if manual_split==True this must be the validation LfwImagesPairsDataset
            manual_split: if manual_split==True this must be the test LfwImagesPairsDataset
        """
        super().__init__()
        self.batch_size = batch_size
        self.splitrate = 0.2
        self.dataset = dataset
        self.splitting_points = splitting_points
        self.num_workers = num_workers
        self.train_dataset = None
        self.shuffle = shuffle
        self.val_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.manual_split = manual_split
        self.input_size = input_size

    def setup(self, stage=None):
        # transforms
        transform = transforms.Compose([
            #FaceAlignTransform(shape=self.input_size, kind=FaceAlignTransform.SIMPLE),
            transforms.ToTensor(),
            transforms.Resize((self.input_size, self.input_size))
        ])

        self.dataset.set_transform(transform)

        if not self.manual_split:
            # define split point
            valid, test = self.splitting_points
            n_samples = len(self.dataset)
            val_size = int(n_samples * valid)
            test_size = int(n_samples * test)
            split_size = [n_samples - (val_size + test_size), val_size, test_size]
            print(split_size)

            # split
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, split_size)
        else:
            self.train_dataset = self.dataset
            self.train_dataset.set_transform(transform)
            self.val_dataset.set_transform(transform)
            self.test_dataset.set_transform(transform)

    # return the dataloader for each split
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=self.shuffle,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=None,
                                           collate_fn=None
                                           )


class LfwImagesDataset(Dataset):
    """ Face dataset. """

    def __init__(self, data_map, transform=None):
        """
        Args:
            data_map: key,value map of people and faces
        """
        self.image_map = data_map
        self.idx_encoding = self._idx_people_encode()
        self.seed = seed(len(data_map.keys()))
        self.transform = transform

    def _idx_people_encode(self):
        """Private function used for the index encoding of the dataset"""
        idx_encoding = []
        for key in self.image_map:
            for img_path in self.image_map[key]:
                idx_encoding.append((key, img_path))
        return idx_encoding

    def set_transform(self, transform):
        """Set the transform attribute for image transformation"""
        self.transform = transform

    def __getitem__(self, idx):
        key, path = self.idx_encoding[idx]
        image1 = Image.open(path)
        image2 = image1

        # if index is even pick an adjacent picture
        if idx % 2 == 0:
            label = 0.0
            if idx < len(self.idx_encoding) - 1:
                key2, path2 = self.idx_encoding[idx + 1]
            else:
                key2, path2 = self.idx_encoding[idx - 1]
            image2 = Image.open(path2)
        # if index is odd pick a random picture form the dataset of a different person
        else:
            label = 1.0
            key2 = key
            while key2 == key:
                new_idx = randint(0, len(self.idx_encoding) - 1)
                key2, path2 = self.idx_encoding[new_idx]
                image2 = Image.open(path2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return len(self.idx_encoding)


class LfwImagesPairsDataset(Dataset):
    """ Face dataset. """

    def __init__(self, data_map, pairs, transform=None):
        """
        Args:
            data_map: key,value map of people and faces
            pairs: list of the pairs from the dataser pair.txt file [[v1,v2,v3,v4],...]
        """
        self.image_map = data_map
        self.seed = seed(len(data_map.keys()))
        self.transform = transform
        self.pairs = pairs

    def set_transform(self, transform):
        """Set the transform attribute for image transformation"""
        self.transform = transform

    def __getitem__(self, idx):
        values = self.pairs[idx]

        label = 1.0
        person1 = values[0]
        picture_path1 = self.image_map[person1][int(values[1])-1]
        picture_path2 = ""

        if len(values) == 3:
            label = 0.0
            picture_path2 = self.image_map[person1][int(values[2][:-1])-1]
        else:
            label = 1.0
            picture_path2 = self.image_map[values[2]][int(values[3][:-1])-1]

        image1 = Image.open(picture_path1)
        image2 = Image.open(picture_path2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

    def __len__(self):
        return len(self.pairs)
