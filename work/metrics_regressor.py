import numpy as np
import torch

class ContingencyMetric:
    def __init__(self, thresh: float = 40.):
        self.thresh = thresh
        self.H = 0
        self.M = 0
        self.F = 0
        self.C = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        preds = preds[mask]
        labels = labels[mask]
        thresh = self.thresh

        self.H += ((preds >= thresh) & (labels >= thresh)).sum()
        self.M += ((preds < thresh) & (labels >= thresh)).sum()
        self.F += ((preds >= thresh) & (labels < thresh)).sum()
        self.C += ((preds < thresh) & (labels < thresh)).sum()

    def reset(self):
        self.H = 0
        self.M = 0
        self.F = 0
        self.C = 0

    def compute(self):
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
