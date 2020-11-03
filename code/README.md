# Code Folder

In this folder you will find all necessary files to train and test an binary CNN model for classification.

## Files
* [__dataset.py:__](./dataset.py) This file creates a custom dataset and dataloaders to read and load your own images.
* [__hyperparameters.py:__](./hyperparameters.py) This is where you can modify hyperparameters for your own runs.
* [__metrics.py:__](./metrics.py) File to implement metrics and is used to get training and testing metrics.
* [__models.py:__](./models.py) This file creates the CNN models that you can use.
* [__run_test.py:__](./run_test.py) File to test a trained model, creates a report with metrics results. (Uses [__testing.py__](./testing.py))
* [__run_train.py:__](./run_train.py) This file trains a model and save the weights from the best epoch. (Uses [__training.py__](./training.py))
* [__testing.py:__](./testing.py) Implements all methods for the testing phase.
* [__training.py:__](./training.py) Implements all methods for the training phase.

## Usage
These are the basic steps to run the training and testing cycles for a CNN model.
1. Refer to the [data folder](../data/), first, you should change the [mean and standard deviation values](../data/preprocessing/) in the [hyperparameters](./hyperparameters.py) file. 
Then you should have created the [dataset splits](../data/cross_validation/create_dataset.py) or [K-Fold files](../data/cross_validation/create_dataset_k_fold.py) for your [dataset](../data/cross_validation/).
2. ```python run_train.py``` To run the training of your model. Remember to modify [hyperparameters](./hyperparameters.py) according to your needs.
3. ```python run_test.py``` To run the testing of your model. Check the report on the [results folder](../results/reports/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)