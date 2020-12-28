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
from src.tools.combine_sampler import CombineSampler
from collections import defaultdict
from src.tools.dataset_tools import get_labels
from enum import Enum
import config_celeba

class DATASETS(Enum):
    CELEBA = 1

# TODO: Dataloader is nearly the same for also for lfw so lets move it to a common place
class CelebA_DataModule(pl.LightningDataModule):

    def __init__(self, name=DATASETS.CELEBA, nb_classes=1000, class_split=True, batch_size=32, splitting_points=(0.10, 0.10),
                 num_workers=4, manual_split=False, valid_dataset=None, test_dataset=None, input_shape=(3, 218, 178),
                 num_classes_iter=8, finetune=False):
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
        self.test_dataset = test_dataset
        self.manual_split = manual_split
        self.input_shape = input_shape
        self.num_classes_iter = num_classes_iter
        self.num_elements_class = int(batch_size / num_classes_iter)
        self.nb_classes = nb_classes
        self.class_split = class_split
        self.finetune = finetune
        torch.manual_seed(0)

    def setup(self, stage=None):
        # transforms
        transform = transforms.Compose([
            # FaceAlignTransform(FaceAlignTransform.ROTATION),
            transforms.ToTensor(),
            transforms.Resize((self.input_shape[1], self.input_shape[2]))
        ])

        if self.name == DATASETS.CELEBA:
            labels_map = get_labels(config=config_celeba)
        else:
            raise Exception("Unknow dataset! Please choose a valid dataset name.")

        valid, test = self.splitting_points
        train = 1 - (valid + test)
        if self.class_split:
            nb_classes_train_val = int(self.nb_classes * (train+valid))
            nb_classes_test = int(self.nb_classes * test)

            total = sum([nb_classes_train_val, nb_classes_test])
            diff = abs(self.nb_classes - total)

            if diff != 0:
                nb_classes_test += diff

            total = sum([nb_classes_train_val, nb_classes_test])
            diff = abs(self.nb_classes - total)

            assert diff == 0

            start, end = 0,  nb_classes_train_val
            print('train classes', start, end)

            self.train_val_dataset = CelebADataset(labels_map, num_classes=list(range(end)))

            n_samples = len(self.train_val_dataset)
            val_size = int(n_samples * valid)
            split_size = [n_samples - val_size, val_size]

            start, end = end, end+nb_classes_test
            print('test classes', start, end)
            self.test_dataset = CelebADataset(labels_map, num_classes=list(range(start, end)))

            for i_dataset in [self.train_val_dataset, self.test_dataset]:
                i_dataset.set_transform(transform)

            self.train_dataset, self.val_dataset = random_split(self.train_val_dataset, split_size)
            print('split size', len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))

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

        self.train_list_of_indices_for_each_class = self._get_list_of_indices(self.train_dataset)
        self.val_list_of_indices_for_each_class = self._get_list_of_indices(self.val_dataset)
        self.test_list_of_indices_for_each_class = self._get_list_of_indices(self.test_dataset)


    def _get_list_of_indices(self, dataset):
        ddict = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])

        return list_of_indices_for_each_class

    # return the dataloader for each split
    def train_dataloader(self):

        sampler = None
        if not self.finetune:
            sampler = CombineSampler(
                self.train_list_of_indices_for_each_class,
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
                self.val_list_of_indices_for_each_class,
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

        sampler = None
        if not self.finetune:
            sampler = CombineSampler(
                self.test_list_of_indices_for_each_class,
                int(self.num_classes_iter * 2),
                self.num_elements_class)

        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size * 2,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=sampler,
                                           collate_fn=None
                                           )


class CelebADataset(Dataset):
    """ Face dataset. """

    def __init__(self, data_map, num_classes, transform=None):
        """
        Args:
            data_map: key,value map of people and faces
        """
        self.image_map = data_map
        self.labels = num_classes
        self.ys, self.im_paths = self._idx_people_encode()
        # self.idx_encoding = self._idx_people_encode()
        self.seed = seed(len(data_map.keys()))
        self.transform = transform

    def _idx_people_encode(self):
        """Private function used for the index encoding of the dataset"""
        # idx_encoding = []
        # for key in self.image_map:
        #     for img_path in self.image_map[key]:
        #         idx_encoding.append((key, img_path))
        # return idx_encoding

        ys, im_paths = [], []
        for key in self.image_map:
            for img_path in self.image_map[key]:
                if key-1 in self.labels:
                    ys += [key - 1]
                    im_paths.append(img_path)

        return ys, im_paths

    def set_transform(self, transform):
        """Set the transform attribute for image transformation"""
        self.transform = transform

    def nb_classes(self):
        return len(np.unique(self.ys))

    def __getitem__(self, idx):
        # label, path = self.idx_encoding[idx]
        # image = Image.open(path)
        # if self.transform is not None:
        #     image = self.transform(image)
        # return image, torch.from_numpy(np.array([label], dtype=np.float32))
        im = Image.open(self.im_paths[idx])
        if self.transform is not None:
            im = self.transform(im)

        return im, self.ys[idx]

    def __len__(self):
        return len(self.ys)
