"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Hyperparameters for a run.
"""


parameters = {
    # Random Seed
    'seed': 123,

    # Data
    'train_file': '../data/train_labels.csv', # Path to the training dataset csv
    'val_file': '../data/val_labels.csv', # Path to the validation dataset csv
    'test_file': '../data/test_labels.csv', # Path to the testing dataset csv
    'k_fold_files': '../data/k_fold/', # Path to the K-Fold splits csvs

    'img_size': 224, # Image input size (this might change depending on the model)
    'batch_size': 32, # Input batch size for training (you can change this depending on your GPU ram)
    'data_mean': [0.485, 0.456, 0.406], # Mean values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)
    'data_std': [0.229, 0.224, 0.225], # Std Dev values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)

    # Model
    'model': 'efficientnet', # Model to train (This name has to correspond to a model from models.py)
    'optimizer': 'ADAM', # Optimizer to update model weights (Currently supported: ADAM or SGD)
    'lear_rate': 0.0001, # Learning Rate to use
    'epochs': 30, # Number of epochs to train for

    # Saving Weights
    'save_name': 'run_efficientnet', # Name of the file to save the trained weights
    'save_criteria': 'F1', # Metric to use for saving the best weights (Currently supported: F1, Accuracy, Loss, Sensitivity, Specificity)
    'weights_path': '../results/weights/', # Path to save the weights

    # Saving Testing Results
    'report_path': '../results/reports/' # Path to save the test report
}