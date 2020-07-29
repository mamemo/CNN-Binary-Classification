''' File to control flow of the program '''

import random
import os
import pathlib
import numpy as np

import torch
import torch.nn as nn

from hyperparameters import parameters as params
from models import resnet18
from dataset import get_aug_dataloader, get_dataloader
from training import train_validate


def seed_everything(seed):
    ''' Set random seed on all environments '''
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    ''' Main function, flow of program '''

    # To stablish a seed for all the project
    seed_everything(parms['seed'])

    # Model
    model = resnet18()

    # Running architecture (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Images Loaders
    train_loader = get_aug_dataloader(params['train_file'], params['img_size'],\
            params['batch_size'], params['data_mean'], params['data_std'])
    val_loader = get_dataloader(params['val_file'], params['img_size'],\
            params['batch_size'], params['data_mean'], params['data_std'])

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
    train_validate(model, train_loader, val_loader, optimizer,\
                    criterion, device, params['epochs'], params['save_criteria'],\
                    params['weights_path'])


if __name__ == "__main__":
    main()