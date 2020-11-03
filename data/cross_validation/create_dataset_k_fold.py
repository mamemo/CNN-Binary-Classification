"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to create a the files for K-Fold validation.

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
from sklearn.model_selection import KFold


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
NUMBER_0F_FOLDS = 5

kfolds = KFold(n_splits=NUMBER_0F_FOLDS, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kfolds.split(images)):
    train_x = [images[id] for id in train_ids]
    test_x = [images[id] for id in test_ids]
    train_y = [labels[id] for id in train_ids]
    test_y = [labels[id] for id in test_ids]


    print("Saving dataset "+str(fold)+"...")

    # Save the splits on csv files
    train_df = pd.DataFrame({'ID_IMG':train_x, 'LABEL': train_y})
    train_df.to_csv('../k_fold/train_labels_'+str(fold)+'.csv')

    test_df = pd.DataFrame({'ID_IMG':test_x, 'LABEL': test_y})
    test_df.to_csv('../k_fold/test_labels_'+str(fold)+'.csv')
