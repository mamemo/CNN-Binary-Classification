"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to calculate the mean and standard deviation of a dataset.
"""

import pandas as pd
import cv2
from PIL import Image
from barbar import Bar

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    """
        Defines a Custom Dataset
    """

    def __init__(self, ids, transf):
        """
            init Constructor

            @param self Object.
            @param ids Path to the images.
            @param transf Transformations to apply.
        """
        super().__init__()

        # Transforms
        self.transforms = transf

        # Images IDS
        self.ids = ids

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
        img = self.transforms(img)

        return img

    def __len__(self):
        return self.data_len


def read_dataset(dir_img):
    """
        read_dataset Read the dataset from a csv file.

        @param dir_img Path to the csv file.
    """

    images = pd.read_csv(dir_img)

    ids = images['ID_IMG'].tolist()
    return ids


def get_dataloader(data_file, img_size, batch_size):
    """
        get_dataloader Creates and returns a dataloader with no data augmentation.

        @param data_file Path to the images csv.
        @param img_size Input size of the model.
        @param batch_size Size of the batch to feed the model with.
    """

    # Read the dataset
    ids = read_dataset(data_file)

    # Transforms
    test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize([img_size]*2, Image.BICUBIC),
        transforms.ToTensor()
    ])

    print("Training Dataset Size: ", len(ids))

    # Create the dataset
    dataset = CustomDataset(ids=ids, transf=test_transform)

    # Create the loaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def get_mean_and_std(dir_img):
    """
        get_mean_and_std Compute the mean and std value of dataset.

        @param dir_img Path to the csv file.
    """

    # Get dataloader
    trainloader = get_dataloader(data_file=dir_img, img_size=224, batch_size=128)

    nimages = 0
    mean = 0.0
    var = 0.0
    for batch in Bar(trainloader):

        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)

        # Update total number of images
        nimages += batch.size(0)

        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print("Dataset Mean: " + str(mean))
    print("Dataset Std Dev: " + str(std))

    return mean, std

if __name__ == "__main__":
    get_mean_and_std(dir_img="../train_labels.csv")
