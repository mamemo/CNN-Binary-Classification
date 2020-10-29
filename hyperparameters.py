"""
    Hyperparameters for a run.
"""


parameters = {
    # Random Seed
    'seed': 123,

    # Data
    'train_file': './dataset_splits/train_labels.csv',
    'val_file': './dataset_splits/val_labels.csv',
    'test_file': './dataset_splits/test_labels.csv',
    
    'img_size': 224,
    'batch_size': 32,
    'data_mean': [0.4815, 0.4815, 0.4815],
    'data_std': [0.2235, 0.2235, 0.2235],

    # Model
    'optimizer': 'ADAM', #or SGD
    'lear_rate': 0.001,
    'epochs': 25,

    # Saving Weights
    'save_criteria': 'Accuracy', #or Loss, Sensitivity, Specificity
    'weights_path': './weights/',

    # Saving Testing Results
    'report_path': './reports/'
}

    # Data
    'train_file': '../data/train_labels.csv',
    'val_file': '../data/val_labels.csv',
    'test_file': '../data/test_labels.csv',
    'k_fold_files': '../data/k_fold/',

    'img_size': 224,
    'batch_size': 32,
    'test_batch_size': 1,
    'data_mean': [0.1664, 0.1708, 0.1751], #[0.1082, 0.1111, 0.1133],
    'data_std': [0.1488, 0.1528, 0.1530], #[0.1512, 0.1546, 0.1548],

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