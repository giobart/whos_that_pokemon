import pytorch_lightning as pl
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from random import seed
from PIL import Image
import numpy as np
from torch.utils.data import random_split
from src.tools.combine_sampler import CombineSampler
from collections import defaultdict
from src.tools.dataset_tools import get_labels, get_dataset_filename_map, get_list_of_indices, get_transforms
from enum import Enum
import config_celeba
import config_lfw

class DATASETS(Enum):
    CELEBA = 1,
    LFW = 2,

class Classification_Model(pl.LightningDataModule):
    def __init__(self, name=DATASETS.CELEBA, nb_classes=1000, class_split=True, batch_size=32, splitting_points=(0.10, 0.10),
                 num_workers=4, manual_split=False, valid_dataset=None, input_shape=(3, 218, 178),
                 num_classes_iter=8, finetune=False, in_folders=False):
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
        self.name = name
        self.splitting_points = splitting_points
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = valid_dataset
        self.manual_split = manual_split
        self.input_shape = input_shape
        self.num_classes_iter = num_classes_iter
        self.num_elements_class = int(batch_size / num_classes_iter)
        self.nb_classes = nb_classes
        self.class_split = class_split
        self.finetune = finetune
        self.in_folders = in_folders
        torch.manual_seed(0)

    def setup(self, stage=None):
        # transforms
        # transform = transforms.Compose([
        #     # FaceAlignTransform(FaceAlignTransform.ROTATION),
        #     transforms.ToTensor(),
        #     transforms.Resize((self.input_shape[1], self.input_shape[2]))
        # ])
        transform = get_transforms(self.input_shape)

        if self.name == DATASETS.CELEBA:
            labels_map = get_labels(config=config_celeba, in_folders=self.in_folders)
        elif self.name == DATASETS.LFW:
            labels_map = get_dataset_filename_map(config = config_lfw)
        else:
            raise Exception("Unknow dataset! Please choose a valid dataset name.")

        valid, test = self.splitting_points
        train = 1 - (valid + test)
        if self.class_split:
            nb_classes_train_val = int(self.nb_classes * (train+valid))
            nb_classes_test = int(self.nb_classes * test)
            self.nb_classes_test = nb_classes_test

            total = sum([nb_classes_train_val, nb_classes_test])
            diff = abs(self.nb_classes - total)

            if diff != 0:
                nb_classes_train_val += diff

            total = sum([nb_classes_train_val, nb_classes_test])
            diff = abs(self.nb_classes - total)

            assert diff == 0

            start, end = 0,  nb_classes_train_val
            print('train classes', start, end)

            self.train_val_dataset = ClassificationDataset(labels_map, num_classes=list(range(end)))
            self.train_val_dataset.set_transform(transform)

            n_samples = len(self.train_val_dataset)
            val_size = int(n_samples * valid)
            split_size = [n_samples - val_size, val_size]

            self.train_dataset, self.val_dataset = random_split(self.train_val_dataset, split_size)

            self.test_dataset = None
            if nb_classes_test > 0:
                start, end = end, end+nb_classes_test
                print('test classes', start, end)
                self.test_dataset = ClassificationDataset(labels_map, num_classes=list(range(start, end)))
                self.test_dataset.set_transform(transform)
                print('split size', len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
            else:
                print('split size', len(self.train_dataset), len(self.val_dataset))

        elif not self.manual_split:
            # define split point
            self.dataset.set_transform(transform)
            n_samples = len(self.dataset)
            val_size = int(n_samples * valid)
            test_size = int(n_samples * test)
            split_size = [n_samples - (val_size + test_size), val_size, test_size]
            print('split size', split_size)
            # split
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, split_size)
        else:
            self.train_dataset = self.dataset
            self.train_dataset.set_transform(transform)
            self.val_dataset.set_transform(transform)
            self.test_dataset.set_transform(transform)


    # return the dataloader for each split
    def train_dataloader(self):

        sampler = None
        if not self.finetune:
            sampler = CombineSampler(
                get_list_of_indices(self.train_dataset),
                self.num_classes_iter,
                self.num_elements_class)

        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=sampler,
                                           collate_fn=None
                                           )

    def val_dataloader(self):

        sampler = None
        if not self.finetune:
            sampler = CombineSampler(
                get_list_of_indices(self.val_dataset),
                int(self.num_classes_iter * 2),
                self.num_elements_class)

        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size * 2,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=sampler,
                                           collate_fn=None
                                           )

    def test_dataloader(self):
        if self.nb_classes_test == 0:
            return None

        sampler = None
        if not self.finetune:
            sampler = CombineSampler(
                get_list_of_indices(self.test_dataset),
                int(self.num_classes_iter * 2),
                self.num_elements_class)

        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size * 2,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=sampler,
                                           collate_fn=None
                                           )

class ClassificationDataset(Dataset):
    """ Face dataset. """

    def __init__(self, data_map, num_classes, transform=None, map_to_int=False, offset_y=1):
        """
        Args:
            data_map: key,value map of people and faces
        """
        self.offset_y = offset_y
        self.image_map = data_map
        self.labels = num_classes
        self.map_to_int = map_to_int
        if map_to_int:
            self.class_to_idx = dict()
            self.encode_classes()
        self.ys, self.im_paths = self._idx_people_encode()
        self.seed = seed(len(data_map.keys()))
        self.transform = transform

    def encode_classes(self):
        for label in list(self.image_map.keys()):
            self.class_to_idx[label] = self.class_to_idx.get(label, len(self.class_to_idx))

    def _idx_people_encode(self):
        """Private function used for the index encoding of the dataset"""
        ys, im_paths = [], []
        for key, value in list(self.image_map.items()):
            for img_path in self.image_map[key]:
                label_id = self.class_to_idx[key] if self.map_to_int else key

                if label_id - self.offset_y in self.labels:
                    ys += [label_id - self.offset_y]
                    im_paths.append(img_path)

        return ys, im_paths

    def set_transform(self, transform):
        """Set the transform attribute for image transformation"""
        self.transform = transform

    def nb_classes(self):
        return len(np.unique(self.ys))

    def __getitem__(self, idx):
        im = Image.open(self.im_paths[idx])
        if self.transform is not None:
            im = self.transform(im)

        return im, self.ys[idx]

    def __len__(self):
        return len(self.ys)
