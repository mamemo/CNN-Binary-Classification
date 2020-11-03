"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to run testing metrics once the model has trained.
"""


import glob
import pathlib

import torch
from torch import nn

from hyperparameters import parameters as params
import models
from dataset import get_dataloader
from testing import test_report


def main():
    """
        main Main function, flow of program.
    """

    # Model
    model = eval('models.' + params['model'] + '()')

    # Running architecture (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using GPU?: ', torch.cuda.is_available())

    # Image loader
    test_loader = get_dataloader(data_file=params['test_file'], img_size=params['img_size'],\
            batch_size=1, data_mean=params['data_mean'], data_std=params['data_std'],\
            data_split='Testing')

    # Creates the criterion (loss function)
    criterion = nn.CrossEntropyLoss()

    # Weights Load Up
    weights_file = glob.glob(params['weights_path']+'/'+params['save_name']+'*.pth')[0]

    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model Loaded!\nAccuracy: {:.4}\nLoss: {:.4}\nSensitivity: {:.4}\nSpecificity: {:.4}'\
            .format(checkpoint['accuracy'], checkpoint['loss'],\
                    checkpoint['sensitivity'], checkpoint['specificity']))


    # Create folder for weights
    pathlib.Path(params['report_path']).mkdir(parents=True, exist_ok=True)


    # Run test metrics and creates a report
    test_report(model=model, dataloader=test_loader, criterion=criterion,\
                device=device, report_path=params['report_path'],\
                save_name=params['save_name'])


if __name__ == "__main__":
    main()
