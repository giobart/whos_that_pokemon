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
    def __init__(self, dataset, batch_size=32, splitting_points=(0.11, 0.11), num_workers=4):
        """
        Args:
            dataset: LfwImagesDataset()
            batch_size: default value: 32
            splitting_points:   splitting point % for train, test and validation.
                                default (0.6,0.85) -> 60% train, 25% validation, 15% test
        """
        super().__init__()
        self.batch_size = batch_size
        self.splitrate = 0.2
        self.dataset = dataset
        self.splitting_points = splitting_points
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        # transforms
        transform = transforms.Compose([
            FaceAlignTransform(FaceAlignTransform.AFFINE),
            transforms.ToTensor(),
        ])

        self.dataset.set_transform(transform)

        # define split point
        valid, test = self.splitting_points
        n_samples = len(self.dataset)
        val_size = int(n_samples * valid)
        test_size = int(n_samples * test)
        split_size = [n_samples - (val_size + test_size), val_size, test_size]
        print(split_size)

        # split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, split_size)

    # return the dataloader for each split
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           shuffle=True,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           shuffle=True,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           shuffle=True,
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
            if idx < len(self.idx_encoding) - 1:
                key2, path2 = self.idx_encoding[idx + 1]
            else:
                key2, path2 = self.idx_encoding[idx - 1]
            image2 = Image.open(path2)
        # if index is odd pick a random picture form the dataset of a different person
        else:
            key2 = key
            while key2 == key:
                new_idx = randint(0, len(self.idx_encoding) - 1)
                key2, path2 = self.idx_encoding[new_idx]
                image2 = Image.open(path2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, key == key2

    def __len__(self):
        return len(self.idx_encoding)
