''' Applies the testing cycle on a data loader. Gets testing metrics. ''' 

import torch
from torch.functional import F
import pandas as pd

from metrics import Metrics


def test(model, dataloader, criterion, device):
    ''' Runs test for a data loader. '''

    # Prepare the model
    model.to(device)
    model.eval()

    # Create storage variables
    metrics = Metrics()
    ids = []
    labels = []
    preds = []

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
            ids.append(id_imgs[0])
            labels.append(labels.item())
            preds.append(predicted.item())

    metrics.print_summary()
    return ids, labels, preds, metrics.summary()


def test_report(model, dataloader, criterion, device, report_path):
    ''' Runs testing and creates a CSV report. '''

    ids, labels, preds, metrics = test(model, dataloader, criterion, device)
    
    test_df = pd.DataFrame(columns=['IDs', 'Labels', 'Pred'])

    for i in range(len(ids)):
        test_df.loc[i] = [ids[i], labels[i], preds[i]]

    metrics_df = pd.DataFrame.from_dict(metrics)

    report = pd.concat([test_df, metrics_df], axis=1, sort=False)

    report.to_csv(report_path)