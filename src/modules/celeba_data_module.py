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


# TODO: Dataloader is nearly the same for also for lfw so lets move it to a common place
class CelebA_DataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size=32, splitting_points=(0.10, 0.10), num_workers=4,
                 manual_split=False, valid_dataset=None, test_dataset=None, input_shape=(3, 218, 178),
                 num_classes_iter=8):
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
        self.val_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.manual_split = manual_split
        self.input_shape = input_shape
        self.num_classes_iter = num_classes_iter
        self.num_elements_class = int(batch_size / num_classes_iter)
        torch.manual_seed(0)

    def setup(self, stage=None):
        # transforms
        transform = transforms.Compose([
            # FaceAlignTransform(FaceAlignTransform.ROTATION),
            transforms.ToTensor(),
            transforms.Resize((self.input_shape[1], self.input_shape[2]))
        ])

        self.dataset.set_transform(transform)

        if not self.manual_split:
            # define split point
            valid, test = self.splitting_points
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
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=CombineSampler(
                                               self.train_list_of_indices_for_each_class,
                                               self.num_classes_iter,
                                               self.num_elements_class),
                                           collate_fn=None
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size * 2,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size * 2,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=None,
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
        self.labels = list(range(num_classes))
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
                if key in self.labels:
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
