"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Applies the testing cycle on a dataloader.
    It also gets testing metrics.
"""

import torch
# from torch.functional import F
import pandas as pd

from metrics import Metrics


def test(model, dataloader, criterion, device):
    """
        test Runs test for a dataloader.

        @param model Model to use for testing.
        @param dataloader Dataloader with testing images.
        @param criterion Loss criterion.
        @param device Use of GPU.
    """

    # Prepare the model
    model.to(device)
    model.eval()

    # Create storage variables
    metrics = Metrics()
    ids_cont = []
    labels_cont = []
    preds_cont = []

    # Test
    with torch.no_grad():
        for id_imgs, inputs, labels in dataloader:

            print(id_imgs[0])

            # Transforming inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # Get predictions
            outputs = model(inputs)
            # probs = F.softmax(outputs, dim=1).data.cpu().numpy()[0]
            _, predicted = torch.max(outputs.data, 1)

            # Get loss
            loss = criterion(outputs, labels)

            # Register on metrics
            metrics.batch(labels=labels, preds=predicted, loss=loss.item())

            # Logs
            ids_cont.append(id_imgs[0])
            labels_cont.append(labels.item())
            preds_cont.append(predicted.item())

    # Print metrics
    metrics.print_summary()
    return ids_cont, labels_cont, preds_cont, metrics.summary()


def test_report(model, dataloader, criterion, device, report_path, save_name):
    """
        test_report Runs testing and creates a CSV report.

        @param model Model to use for testing.
        @param dataloader Dataloader with testing images.
        @param criterion Loss criterion.
        @param device Use of GPU.
        @param report_path Path to store the results report.
        @param save_name Name to save the report with.
    """

    # Test model
    ids, labels, preds, metrics = test(model, dataloader, criterion, device)

    # Extra metrics (after getting testing predictions)
    metrics['Model AUC'] = Metrics.auc(labels, preds)
    print('Model AUC: ', metrics['Model AUC'])
    metrics['Model 95% CI'] = Metrics.ci(labels, preds)
    print('Model 95% CI: ', metrics['Model 95% CI'])

    # Dataframe to store predictions
    test_df = pd.DataFrame(columns=['IDs', 'Labels', 'Pred'])

    # Load up dataframe
    for i in range(len(ids)):
        test_df.loc[i] = [ids[i], labels[i], preds[i]]

    # Create metrics dataframe
    metrics_df = pd.DataFrame.from_dict(metrics)

    # Merge predictions and metrics dataframes
    report = pd.concat([test_df, metrics_df], axis=1, sort=False)

    # Save final dataframe
    report.to_csv(report_path + save_name + '.csv')
