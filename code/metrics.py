"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Implementation of metrics and results accumulator.

    * Supported Metrics and Accumulators (on training)
        - Accuracy
        - Loss
        - Sensitivity
        - Specificity
        - F1
        - Precision

    * After getting preds and labels all together (on testing)
        - Area under the curve (AUC)
        - 95% Confidence Interval (CI)
"""


from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint
import numpy as np


class Metrics():
    """
        Class to calculate metrics over training and testing.
    """

    def __init__(self):
        """
            init Constructor method.

            @param self Created object.
        """

        # * Accumulators for Metrics
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0

        self.loss = 0
        self.batches = 0
        self.labels = []
        self.preds = []


    def batch(self, labels, preds, loss=0):
        """
            batch Method to update metrics acummulators for every batch of
                    processed images.

            @param self Object.
            @param labels Respective labels of the processed images.
            @param preds Predicted outputs.
            @param loss Loss of predictions over labels.
        """
        self.true_pos += ((labels == preds) & (labels == 1)).sum().item()
        self.true_neg += ((labels == preds) & (labels == 0)).sum().item()
        self.false_pos += ((labels != preds) & (labels == 0)).sum().item()
        self.false_neg += ((labels != preds) & (labels == 1)).sum().item()

        self.loss += loss
        self.batches += 1


    def summary(self):
        """
            summary Returns a summary of the metrics results.

            @param self Object.
        """
        return {
            "Model Accuracy":     [self.accuracy(self.true_pos, self.true_neg,\
                                    self.false_pos, self.false_neg)],
            "Model Loss":         [self.loss],
            "Model Sensitivity":  [self.sensitivity(self.true_pos, self.false_neg)],
            "Model Specificity":  [self.specificity(self.true_neg, self.false_pos)],
            "Model F1":           [self.f1(self.true_pos, self.false_pos, self.false_neg)],
            "Model Precision":    [self.precision(self.true_pos, self.false_pos)]
        }


    def print_summary(self):
        """
            print_summary Prints the summary of the metrics results.

            @param self Object.
        """
        summ = self.summary()
        for key in summ:
            print(key+": ", summ[key])


    def print_one_liner(self, phase='Train'):
        """
            print_one_liner Prints a quick summary of the metrics results in one line.

            @param self Object.
            @param phase Respective run phase where the function is called from.
        """
        summ = self.summary()
        print('{} Acc: {:.4}, {} Loss: {:.4}, {} Sens: {:.4}, {} Spec: {:.4} '\
            .format(phase, summ["Model Accuracy"][0], phase, summ["Model Loss"][0], \
                    phase, summ["Model Sensitivity"][0], phase, summ["Model Specificity"][0]))
        return summ


    def accuracy(self, true_pos, true_neg, false_pos, false_neg):
        """
            accuracy Measures the accuracy of the accumulated predictions.

            @param self Object.
            @param true_pos True positives.
            @param true_neg True negatives.
            @param false_pos False positives.
            @param false_neg False negatives.
        """
        if true_pos + true_neg + false_pos + false_neg == 0:
            return 0
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)


    def sensitivity(self, true_pos, false_neg):
        """
            sensitivity Measures the sensitivity of the accumulated predictions.

            @param self Object.
            @param true_pos True positives.
            @param false_neg False negatives.
        """
        if true_pos + false_neg == 0:
            return 0
        return (true_pos) / (true_pos + false_neg)


    def specificity(self, true_neg, false_pos):
        """
            specificity Measures the specificity of the accumulated predictions.

            @param self Object.
            @param true_neg True negatives.
            @param false_pos False positives.
        """
        if true_neg + false_pos == 0:
            return 0
        return (true_neg) / (true_neg + false_pos)


    def precision(self, true_pos, false_pos):
        """
            precision Measures the precision of the accumulated predictions.

            @param self Object.
            @param true_pos True positives.
            @param false_pos False positives.
        """
        if true_pos + false_pos == 0:
            return 0
        return (true_pos) / (true_pos + false_pos)


    def f1(self, true_pos, false_pos, false_neg):
        """
            f1 Measures the f1 of the accumulated predictions.

            @param self Object.
            @param true_pos True positives.
            @param false_pos False positives.
            @param false_neg False negatives.
        """
        if 2 * true_pos + false_pos + false_neg == 0:
            return 0
        return (2 * true_pos) / (2 * true_pos + false_pos + false_neg)


    @staticmethod
    def auc(labels, preds):
        """
            auc Measures the AUC of the accumulated predictions.

            @param self Object.
            @param labels Respective labels of the processed images.
            @param preds Predicted outputs
        """
        return roc_auc_score(labels, preds)


    @staticmethod
    def ci(labels, preds):
        """
            ci Measures the 95% CI of the accumulated predictions.

            @param self Object.
            @param labels Respective labels of the processed images.
            @param preds Predicted outputs
        """
        correct = len(labels) - sum(abs(np.array(labels) - np.array(preds)))
        lower, upper = proportion_confint(correct, len(labels), 0.05)
        return f'{lower:.4} - {upper:.4}'
