# CNN Binary Classification Template

This project is meant to work as a template for a binary CNN classification problem. 
You are encourage to use this code as a base for your project, modifying it when it's necessary.

## Installation

Before you clone this repo first do the following:

1. Click on __Use this template__. This will create you a brand new repository using this one as a template, allowing you to modify it as you please and commit those changes. Otherwise you won't be able to commit any changes unless it's a contribution, in that case, please submit a pull request.
2. Once you forked the repo, go to your account and clone the repo as you would normally.
3. ```cd PATH/CNN-Binary-Classification/``` In a terminal go to the project folder.
4. ```pip install -r requeriments.txt``` Install all requirements. (Note: Pytorch version might need to installed separately. For more installing info visit [__pytorch.org__](https://pytorch.org/))
5. Change the code and have fun!

## Folders
* [__code/__](./code/) This folder contains the files to train and test a model.
* [__data/__](./data/) Here you will find files to create dataset csvs and explore your own dataset.
* [__results/__](./results/) This folder contains the best weights and testing reports for the models you will train.

## Recommendations
* On a project you should at least run your training 3 times:
    1. Using a single time split (train/val/test): Here you will experiment and play with the model, hyperparameters, optimizer, loss, etc.
    2. Using K-Fold validation: Once you are happy with the performance of your approach, it's time to validate it. We use validation to assess how the model perform with new data, in other words how good your model is at generalizing. (Note: The average of the folds is what you should report as your model performance.)
    3. Using the whole: And finally, if you are happy with how your model do with new data, it's time to train the last model, the model to use on production. For this, you have to train the same model and hyperparameters as before but with the whole data, giving it the best chance to perform well on a real scenario.
* When working on any python project you should create a virtual environment, this allows to install and remove any package without having version conflicts. To create a virtual environment run ```virtualenv --python=python3 venv``` then use go into it run ```source PATH/venv/bin/activate``` and finally if you want to exit run ```deactivate```. (Note: These commands might change for Windows.)
* Sometimes you might be in the situation where you are running scripts on a multi-GPU computer but you want to use an specific GPU, for that case you can use ```CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python...``` (Note: Change ```0``` for the number of the desired GPU.)

## Future work
1. Parallel training: Support for training on multiple GPUs.
2. Multiclass classification: Extending the code for multiple classes instead of binary.
3. More hyperparameters: Give more options to you (optimizers, losses, metrics, models).

## Author:
* [__Mauro Mendez__](https://github.com/mamemo/)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
