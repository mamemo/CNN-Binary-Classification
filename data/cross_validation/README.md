# Cross Validation Folder

In this folder you will find the files to apply cross validation over your dataset. Currently single time split (train/val/test) and K-fold validation are supported.

## Files
* [__amount_data.py:__](./amount_data.py) This file allows you to count the images in your dataset splits.
* [__create_dataset.py:__](./create_dataset.py) Here you will create the csvs file for the three dataset splits (train/val/test). You should change the code according to your own dataset.
* [__create_dataset_k_fold.py:__](./create_dataset_k_fold.py) This file creates the csvs files for k-fold validation. You should change the code according to your own dataset.

## Usage
To create the csvs files for a validation you should run:
1. ```python create_dataset.py``` or ```python create_dataset_k_fold.py``` This will look into a folder containig the data and create the csvs the model is going to use for training and testing.
2. ```python amount_data.py``` In case you want to make sure how many images are in total within each split.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
