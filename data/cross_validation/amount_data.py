"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to count the images and cases on the dataset.
"""

import pandas as pd


# Read dataset csvs
train_dataset = pd.read_csv('../train_labels.csv')
val_dataset = pd.read_csv('../val_labels.csv')
test_dataset = pd.read_csv('../test_labels.csv')


# Get numbers for training dataset
imgs_train = train_dataset['ID_IMG']
pos_train_imgs = train_dataset[train_dataset['LABEL']==1]

# Get numbers for validation dataset
imgs_val = val_dataset['ID_IMG']
pos_val_imgs = val_dataset[val_dataset['LABEL']==1]

# Get numbers for testing dataset
imgs_test = test_dataset['ID_IMG']
pos_test_imgs = test_dataset[test_dataset['LABEL']==1]


# Print results
print('\nTraining Images: ', len(imgs_train))
print('Positive Images: ', len(pos_train_imgs))

print('\nValidation Images: ', len(imgs_val))
print('Positive Images: ', len(pos_val_imgs))

print('\nTesting Images: ', len(imgs_test))
print('Positive Images: ', len(pos_test_imgs))

print('\nTotal Images: ', len(imgs_train)+len(imgs_test)+len(imgs_val))
print('Total Positive Images: ', len(pos_train_imgs)+len(pos_val_imgs)+len(pos_test_imgs))
