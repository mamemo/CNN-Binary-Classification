# Preprocessing Folder

In this folder you will find scripts to load, play and experiment with different preprocessing techniques.

## Files
* [__mean_std_dataset.py:__](./mean_std_dataset.py) Gets the mean and standard deviation values for your dataset. This values are indispensable and must be registered on [__hyperparameters.py__](../../code/hyperparameters.py).

## Usage
To get the mean and standard deviation values for your dataset you must:
1. Create the dataset splits using [__cross_validation/__](../cross_validation/).
2. ```python mean_std_dataset.py``` To get the values.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
