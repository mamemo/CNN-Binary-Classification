"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to create the dataset and dataloaders.
"""

import pandas as pd
from PIL import Image
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from imgaug import augmenters as iaa


class CustomDataset(Dataset):
    """
        Defines a Custom Dataset
    """
    
    def __init__(self, ids, labels, transf):
        """
            init Constructor

            @param self Object.
            @param ids Path to the images.
            @param labels Labels of the images.
            @param transf Transformations to apply.
        """
        super().__init__()

        # Transforms
        self.transforms = transf

        # Images IDS amd Labels
        self.ids = ids
        self.labels = torch.LongTensor(labels)

        # Calculate len of data
        self.data_len = len(self.ids)

    def __getitem__(self, index):
        """
            getitem Method to get one image.

            @param self Object.
            @param index The position of the image in the dataset.
        """
        # Get an ID of a specific image
        id_img = self.ids[index]

        # Open Image
        img = cv2.imread(id_img)
        img = cv2.convertScaleAbs(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        # Applies transformations
        if self.transforms:
            img = self.transforms(img)

        # Get Label
        label = self.labels[index]

        return (id_img, img, label)

    def __len__(self):
        return self.data_len


class ImgAugTransform(object):
    """
        Class to define the transformations to apply to the images.
    """
    def __init__(self):
        """
            init Constructor

            @param self Object.
        """
        self.aug = iaa.Sequential([
            # Blur or Sharpness
            iaa.Sometimes(0.25,
                            iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
                                     iaa.pillike.EnhanceSharpness(factor=(0.8,1.5))])),
            # Flip horizontally
            iaa.Fliplr(0.5),
            # Rotation
            iaa.Rotate((-20, 20)),
            # Pixel Dropout
            iaa.Sometimes(0.25,
                            iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            # Color
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
        ])
    def __call__(self, img):
        """
            call What to do when applied to an image

            @param self Object.
            @param img Image to work on.
        """
        img = np.array(img)
        return self.aug.augment_image(img)


def read_dataset(dir_img):
    """
        read_dataset Read the dataset from a csv file.

        @param dir_img Path to the csv file
    """

    images = pd.read_csv(dir_img)
    ids = images['ID_IMG'].tolist()
    labels = images['LABEL'].tolist()
    return ids, labels


def get_aug_dataloader(train_file, img_size, batch_size, data_mean, data_std):
    """
        get_aug_dataloader Creates and return a dataloader with data augmentation.

        @param train_file Path to the training images csv.
        @param img_size Input size of the model.
        @param batch_size Size of the batch to feed the model with.
        @param data_mean Mean values of the dataset (for normalization).
        @param data_std Standard deviation values of the dataset (for normalization).
    """
    
    # Read the dataset
    ids, labels = read_dataset(train_file)

    #Transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize([img_size]*2, Image.BICUBIC),

        # Augmentation
        ImgAugTransform(),

        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) # Change for every dataset
    ])

    print("Training Dataset Size: ", len(ids))

    # Create the dataset
    train_dataset = CustomDataset(ids=ids, labels=labels, transf=train_transform)

    # Create the loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_dataloader(data_file, img_size, batch_size, data_mean, data_std, data_split = 'Validation'):
    """
        get_dataloader Creates and returns a dataloader with no data augmentation.

        @param data_file Path to the images csv.
        @param img_size Input size of the model.
        @param batch_size Size of the batch to feed the model with.
        @param data_mean Mean values of the dataset (for normalization).
        @param data_std Standard deviation values of the dataset (for normalization).
        @param data_split Training process where this is dataloader is used (Val or Test).
    """

    # Read the dataset
    ids, labels = read_dataset(data_file)

    # Transformations
    test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize([img_size]*2, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
    ])

    print(data_split+" Dataset Size: ", len(ids))

    # Create the dataset
    dataset = CustomDataset(ids=ids, labels=labels, transf=test_transform)

    # Create the loaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader
