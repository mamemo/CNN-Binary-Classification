parameters = {
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