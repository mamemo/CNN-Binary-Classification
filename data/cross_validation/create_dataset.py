"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to create a single time split (training/validation/testing).

    The dataset is expected to be in a folder following the structure:

    data/
        cross_validation/  (The folder you're currently in)
        dataset/
            0/
            1/
        preprocessing/

    You must change the logic to read your dataset in case it follows another structure.
    The bottom section of this code expects a list with the absoulte path to the images
    and a list with their labels.
"""

import glob
import pandas as pd
from sklearn.model_selection import train_test_split


#! /////////// Change code to read your dataset //////

SPLIT_CHAR = '/' # Change for \\ if you're using Windows
DATASET_FOLDER = '..' + SPLIT_CHAR + 'dataset' + SPLIT_CHAR  # Change '..' for an absolute path
IMAGE_EXTENSION = '*.png' # Change for the extension of your images


print('Reading Dataset...')

# Get absolute paths to all images in dataset
images = glob.glob(DATASET_FOLDER + '*' + SPLIT_CHAR + IMAGE_EXTENSION)

# Get labels per image
labels = [int(img.split(SPLIT_CHAR)[-2]) for img in images]


print("Splitting dataset...")

# Split dataset
train_ratio = 0.75
val_ratio = 0.1
test_ratio = 0.15

train_x, test_x, train_y, test_y = train_test_split(\
                                    images, labels,\
                                    train_size=train_ratio,\
                                    stratify=labels)
val_x, test_x, val_y, test_y = train_test_split(\
                                    test_x, test_y,\
                                    test_size=test_ratio/(test_ratio+val_ratio),\
                                    stratify=test_y)


print("Saving datasets...")

# Save the splits on csv files
dataset_df = pd.DataFrame({'ID_IMG':images, 'LABEL': labels})
dataset_df.to_csv('../full_dataset_labels.csv')

train_df = pd.DataFrame({'ID_IMG':train_x, 'LABEL': train_y})
train_df.to_csv('../train_labels.csv')

val_df = pd.DataFrame({'ID_IMG':val_x, 'LABEL': val_y})
val_df.to_csv('../val_labels.csv')

test_df = pd.DataFrame({'ID_IMG':test_x, 'LABEL': test_y})
test_df.to_csv('../test_labels.csv')
