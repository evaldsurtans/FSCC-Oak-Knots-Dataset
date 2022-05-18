import torch
import numpy as np
from torchnet.meter import meter

EPS = np.finfo(float).eps

class ClassificationMetricMeter(meter.Meter):
    def __init__(self, num_class=2):
        super(ClassificationMetricMeter, self).__init__()
        self.num_class = num_class
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add(self, predict, target):
        if torch.is_tensor(predict):
            predict = predict.view(predict.size(0), -1).cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.view(target.size(0), -1).cpu().squeeze().numpy()

        assert predict.shape == target.shape, 'target and predict shape should match'

        # binary
        if self.num_class <= 2:
            t = predict == target
            f = np.logical_not(t)
            self.tp += np.sum(t * target)
            self.tn += np.sum(t * np.logical_not(target))
            self.fp += np.sum(f * predict)
            self.fn += np.sum(f * np.logical_not(predict))
        # multiclass
        else:
            for c in range(self.num_class):
                tmp_predict = predict == c
                tmp_target = target == c
                t = tmp_predict == tmp_target
                f = np.logical_not(t)
                self.tp += np.sum(t * tmp_target)
                self.tn += np.sum(t * np.logical_not(tmp_target))
                self.fp += np.sum(f * tmp_predict)
                self.fn += np.sum(f * np.logical_not(tmp_predict))

    def value(self):
        '''
        Returns tuple of precision, recall, f1_score
        '''

        precision = (self.tp + EPS) / (self.tp + self.fp + EPS)
        recall = (self.tp + EPS) / (self.tp + self.fn + EPS)
        f1 = (self.tp + EPS) / (self.tp + (self.fp + self.fn) / 2 + EPS)

        return precision, recall, f1
