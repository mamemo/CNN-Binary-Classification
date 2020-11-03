"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    File to implement the training and validation cycles.
"""

import torch
from barbar import Bar

from metrics import Metrics


def train(model, dataloader, optimizer, criterion, device):
    """
        train Runs one epoch of training.

        @param model Model to train.
        @param dataloader Images to train with.
        @param optimizer Optimizer to update weights.
        @param criterion Loss criterion.
        @param device Use of GPU.
    """

    # Prepare the model
    model.to(device)
    model.train()

    # Creates metrics recorder
    metrics = Metrics()

    # Iterates over batches
    for (_, inputs, labels) in Bar(dataloader):

        # Clean gradients in the optimizer
        optimizer.zero_grad()

        # Transforming inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward Pass
        outputs = model(inputs)

        # Get loss
        loss = criterion(outputs, labels)

        # Backward Pass, updates weights and optimizer
        loss.backward()
        optimizer.step()

        # Register on metrics
        _, predicted = torch.max(outputs.data, 1)
        metrics.batch(labels=labels, preds=predicted, loss=loss.item())

    # Print training metrics
    metrics.print_one_liner()
    return metrics.summary()


def validate(model, dataloader, criterion, device):
    """
        validate Runs one epoch of validation.

        @param model Model to train.
        @param dataloader Images to train with.
        @param criterion Loss criterion.
        @param device Use of GPU.
    """

    # Prepare the model
    model.to(device)
    model.eval()

    # Creates metrics recorder
    metrics = Metrics()

    with torch.no_grad():
        # Iterates over batches
        for (_, inputs, labels) in Bar(dataloader):

            # Transforming inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward Pass
            outputs = model(inputs)

            # Get loss
            loss = criterion(outputs, labels)

            # Register on metrics
            _, predicted = torch.max(outputs.data, 1)
            metrics.batch(labels=labels, preds=predicted, loss=loss.item())

    # Print and return validation metrics
    metrics.print_one_liner(phase='Val')
    return metrics.summary()


def train_validate(model, train_loader, val_loader, optimizer,\
                    criterion, device, epochs, save_criteria, weights_path, save_name):
    """ 
        train_validate Trains and validates a model.

        @param model Model to train on.
        @param train_loader Images to train with.
        @param val_loader Images to use for validation.
        @param optimizer Optimizer to update weights.
        @param criterion Loss criterion.
        @param device Use of GPU.
        @param epochs Amount of epochs to train.
        @param save_criteria What metric to use to save best weights.
        @param weights_path Path to the folder to save best weights.
        @param save_name Filename of the best weights.
    """

    # Initial best model values
    best_criteria = 0
    best_model = {}

    # Iterates over total epochs
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}')
        # Train
        metrics = train(model, train_loader, optimizer, criterion, device)
        # Validate
        if val_loader:
            metrics = validate(model, val_loader, criterion, device)

        # Update best model
        if save_criteria == 'Loss': metrics['Model Loss'][0] *= -1 # Change sign of loss
        if epoch == 1 or metrics['Model '+save_criteria][0] >= best_criteria:
            best_criteria = metrics['Model '+save_criteria][0]
            best_model = {'epoch': epoch,\
                'model_state_dict': model.state_dict(),\
                'optimizer_state_dict': optimizer.state_dict(),\
                'accuracy': metrics['Model Accuracy'][0],\
                'loss': metrics["Model Loss"][0],\
                'sensitivity': metrics["Model Sensitivity"][0],\
                'specificity': metrics["Model Specificity"][0]}

    # Save model
    save_path = '{}{}_{}_{:.6}.pth'.format(weights_path, save_name,\
                save_criteria, str(best_criteria).replace('.', '_'))
    torch.save(best_model, save_path)

    return save_path
