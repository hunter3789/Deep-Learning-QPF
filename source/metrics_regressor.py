import numpy as np
import torch

class ContingencyMetric:
    """
    Metric for computing mean IoU and accuracy

    Sample usage:
    >>> batch_size, num_classes = 8, 6
    >>> preds = torch.randint(0, num_classes, (batch_size,))   # (b,)
    >>> labels = torch.randint(0, num_classes, (batch_size,))   # (b,)
    >>> cm = ConfusionMatrix(num_classes=num_classes)
    >>> cm.add(preds, labels)
    >>> # {'iou': 0.125, 'accuracy': 0.25}
    >>> metrics = cm.compute()
    >>> # clear the confusion matrix before the next epoch
    >>> cm.reset()
    """

    def __init__(self, thresh: float = 40.):
        """
        Builds and updates a confusion matrix.

        Args:
            num_classes: number of label classes
        """
        self.thresh = thresh
        self.H = 0
        self.M = 0
        self.F = 0
        self.C = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """
        Updates using predictions and ground truth labels

        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
        """
        preds = preds[mask]
        labels = labels[mask]
        thresh = self.thresh

        self.H += ((preds >= thresh) & (labels >= thresh)).sum()
        self.M += ((preds < thresh) & (labels >= thresh)).sum()
        self.F += ((preds >= thresh) & (labels < thresh)).sum()
        self.C += ((preds < thresh) & (labels < thresh)).sum()

    def reset(self):
        """
        Resets the confusion matrix, should be called before each epoch
        """
        self.H = 0
        self.M = 0
        self.F = 0
        self.C = 0

    def compute(self):
        """
        Computes the mean IoU and accuracy
        """
        #true_pos = self.matrix.diagonal()
        #target = self.target
        H = self.H
        M = self.M
        F = self.F
        C = self.C

        CSI = POD = FAR = FBIAS = 0
        if H+M+F > 0:
            CSI = H / (H+M+F)
        if H+M > 0:
            POD = H / (H+M)
            FBIAS = (H+F) / (H+M)
        if H+F > 0:
            FAR = F / (H+F)

        #C = true_pos[:target].sum()
        #true_pos = self.matrix.diagonal()
        #class_iou = true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)
        #mean_iou = class_iou.mean().item()
        #accuracy = (true_pos.sum() / (self.matrix.sum() + 1e-5)).item()

        return {
            "H": H,
            "M": M,
            "F": F,
            "C": C,
            "CSI": CSI,
            "POD": POD,
            "FAR": FAR,
            "FBIAS": FBIAS,
        }
