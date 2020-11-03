"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to run the training of a model.
"""

import random
import os
import pathlib
import numpy as np

import torch
from torch import nn

from hyperparameters import parameters as params
from models import *
from dataset import get_aug_dataloader, get_dataloader
from training import train_validate


def seed_everything(seed):
    """
        seed_everything Set random seed on all environments.

        @param seed Random seed.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """
        main Main function, flow of program.
    """

    # To stablish a seed for all the project
    seed_everything(params['seed'])

    # Model
    model = eval(params['model']+'()')

    # Running architecture (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using GPU?: ', torch.cuda.is_available())

    # Image Loaders
    train_loader = get_aug_dataloader(train_file=params['train_file'],\
            img_size=params['img_size'], batch_size=params['batch_size'],\
            data_mean=params['data_mean'], data_std=params['data_std'])
    val_loader = get_dataloader(data_file=params['val_file'], img_size=params['img_size'],\
            batch_size=params['batch_size'], data_mean=params['data_mean'],\
            data_std=params['data_std'])

    # Creates the criterion (loss function)
    criterion = nn.CrossEntropyLoss()

    # Creates optimizer (Changes the weights based on loss)
    if params['optimizer'] == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lear_rate'])
    elif params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lear_rate'], momentum = 0.9)

    # Create folder for weights
    pathlib.Path(params['weights_path']).mkdir(parents=True, exist_ok=True)


    # Training and Validation for the model
    train_validate(model=model, train_loader=train_loader, val_loader=val_loader,\
                    optimizer=optimizer, criterion=criterion, device=device,\
                    epochs=params['epochs'], save_criteria=params['save_criteria'],\
                    weights_path=params['weights_path'], save_name=params['save_name'])


if __name__ == "__main__":
    main()
