"""
    Hyperparameters for a run.
"""


parameters = {
    # Random Seed
    'seed': 123,

    # Data
    'train_file': '../data/train_labels.csv',
    'val_file': '../data/val_labels.csv',
    'test_file': '../data/test_labels.csv',
    'k_fold_files': '../data/k_fold/',

    'img_size': 224,
    'batch_size': 32,
    'data_mean': [0.485, 0.456, 0.406],
    'data_std': [0.229, 0.224, 0.225],

    # Model
    'model': 'efficientnet', # or resnet18
    'optimizer': 'ADAM', # or ADAM, SGD
    'lear_rate': 0.0001,
    'epochs': 50,

    # Saving Weights
    'save_name': 'run_efficientnet',
    'save_criteria': 'F1', # or F1, Accuracy, Loss, Sensitivity, Specificity
    'weights_path': '../results/weights/',

    # Saving Testing Results
    'report_path': '../results/reports/'
}