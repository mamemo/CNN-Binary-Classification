''' File to manage all dataset related stuff '''

import pandas as pd
from PIL import Image
import cv2

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    ''' Defines a Custom Dataset '''
    
    def __init__(self, ids, labels, transf):
        super().__init__()

        # Transforms
        self.transforms = transf

        # Images IDS
        self.ids = ids
        self.labels = torch.LongTensor(labels)

        # Calculate len of data
        self.data_len = len(self.ids)

    def __getitem__(self, index):
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

        label = self.labels[index]

        return (id_img, img, label)

    def __len__(self):
        return self.data_len


def read_dataset(dir_img):
    ''' Read the dataset from a csv file. '''
    
    images = pd.read_csv(dir_img)
    ids = images['ID_IMG'].tolist()
    labels = images['LABEL'].tolist()
    return ids, labels


def get_aug_dataloader(train_file, img_size, batch_size, data_mean, data_std):
    ''' Creates and return a data loader with data augmentation '''
    
    # Read the dataset
    ids, labels = read_dataset(train_file)

    #Transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize([img_size]*2, Image.BICUBIC),

        # Augmentation
        transforms.RandomAffine(degrees=180, translate=(0.02, 0.02),
           scale=(0.98, 1.02), shear=2, fillcolor=(0, 0, 0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

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
    ''' Creates and returns a loader without data augmentation '''
    
    # Read the dataset
    ids, labels = read_dataset(data_file)

    # Transforms
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