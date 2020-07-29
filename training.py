''' File that implements the training and validation processes. '''

import torch
from barbar import Bar

from metrics import Metrics


def train(model, dataloader, optimizer, criterion, device):
    ''' Runs one epoch of training for a model. '''

    # Prepare the model
    model.to(device)
    model.train()

    # Creates metrics recorder
    metrics = Metrics()

    # Iterates over batches
    for (id_imgs, inputs, labels) in Bar(dataloader):

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


def validate(model, dataloader, criterion, device):
    ''' Runs one epoch of validation for a model. '''

    # Prepare the model
    model.to(device)
    model.eval()

    # Creates metrics recorder
    metrics = Metrics()

    with torch.no_grad():
        # Iterates over batches
        for (id_imgs, inputs, labels) in Bar(dataloader):

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
    return metrics.print_one_liner()

def train_validate(model, train_loader, val_loader, optimizer,\
                    criterion, device, epochs, save_criteria, weights_path):
    ''' Trains and validates a model. '''

    best_criteria = 0

    # Iterates over total epochs
    for epoch in range(1, epochs+1):

        # Train
        train(model, train_loader, optimizer, criterion, device)
        # Validate
        metrics = validate(model, val_loader, criterion, device)

        # Save best model
        if save_criteria == 'Loss': metrics['Model Loss'] *= -1 # To always look for max when saving
        if epoch == 1 or metrics['Model '+save_criteria] > best_criteria:
            best_criteria = metrics['Model '+save_criteria]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': metrics['Model Accuracy'],
                'loss': metrics["Model Loss"],
                'sensitivity': metrics["Model Sensitivity"],
                'specificity': metrics["Model Specificity"]
            }, '{}weights_epoch_{}_{}_{}.pth'.format(weights_path, epoch,\
                    save_criteria, str(best_criteria).replace('.', '_')))
