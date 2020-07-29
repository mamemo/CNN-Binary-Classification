''' File to run the model once trained '''

import glob
import pathlib

import torch
import torch.nn as nn

from hyperparameters import parameters as params
from models import resnet18
from dataset import get_dataloader
from testing import test_report

def main():
    ''' Main function, flow of program '''

    # Model
    model = resnet18()

    # Running architecture (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image loader
    test_loader = get_dataloader(params['test_file'], params['img_size'],\
            params['batch_size'], params['data_mean'], params['data_std'])

    # Creates the criterion (loss function)
    criterion = nn.CrossEntropyLoss()

    # Weights Load Up
    weights_file = glob.glob(params['weights_path']+'/*.pth')[0]

    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model Loaded!\nAccuracy: {:.4}\nLoss: {:.4}\nSensitivity: {:.4}\nSpecificity: {:.4}'\
            .format(checkpoint['accuracy'], checkpoint['loss'],\
                    checkpoint['sensitivity'], checkpoint['specificity']))


    # Create folder for weights
    pathlib.Path(params['report_path']).mkdir(parents=True, exist_ok=True)


    # Run test and creates a report
    test_report(model, test_loader, criterion, device, params['report_path'])



if __name__ == "__main__":
    main()