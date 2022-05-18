import torch
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

def IoU(predict, target):
    with torch.no_grad():
        predict = predict.view(target.size(0), -1)
        target = target.view(target.size(0), -1)

        smooth = 1e-5
        intersection = torch.sum(predict * target, dim=1) + smooth
        coef = torch.mean(intersection / (torch.sum((predict + target) > 0, dim=1) + smooth))
        return coef


def IoU_each_class(predict, target):
    with torch.no_grad():
        predict = predict.view(target.size(0), -1)
        target = target.view(target.size(0), -1)

        target_class = torch.sum(target, dim=1)
        target_class = torch.clamp_max(target_class, max=1)

        first_class_indexes = torch.where(target_class)
        zero_class_indexes = torch.where(1 - target_class)

        coef = 0
        if torch.sum(target_class) > 0:
            first_intersection = torch.sum(predict[first_class_indexes] * target[first_class_indexes], dim=1)
            first_coef = first_intersection / (torch.sum((predict[first_class_indexes] + target[first_class_indexes]) > 0, dim=1))
            coef = first_coef
        if torch.sum(1 - target_class) > 0:
            zero_intersection = torch.sum((1 - predict[zero_class_indexes]) * (1 - target[zero_class_indexes]), dim=1)
            zero_coef = zero_intersection / (torch.sum((1 - predict[zero_class_indexes] + 1 - target[zero_class_indexes]) > 0, dim=1))
            if torch.sum(target_class) > 0:
                coef = torch.cat((first_coef, zero_coef))
            else:
                coef = zero_coef
        coef = torch.mean(coef)
        return coef


def dice_coeficient(predict, target, reduction='mean'):
    predict = predict.view(target.size(0), -1)
    target = target.view(target.size(0), -1)

    smooth = 1e-5
    intersection = torch.sum(predict * target, dim=1)
    predict_sum = torch.sum(predict, dim=1)
    target_sum = torch.sum(target, dim=1)
    coef = ((2. * intersection) + smooth) / (predict_sum + target_sum + smooth)
    if reduction == 'mean':
        coef = torch.mean(coef)
    return coef


def dice_coeficient_each_class(predict, target, reduction='mean'):
    predict = predict.view(target.size(0), -1)
    target = target.view(target.size(0), -1)

    target_class = torch.sum(target, dim=1)
    target_class = torch.clamp_max(target_class, max=1)

    first_class_indexes = torch.where(target_class)
    zero_class_indexes = torch.where(1 - target_class)

    coef = 0
    if torch.sum(target_class) > 0:
        first_intersection = torch.sum(predict[first_class_indexes] * target[first_class_indexes], dim=1)
        first_predict_sum = torch.sum(predict[first_class_indexes], dim=1)
        first_target_sum = torch.sum(target[first_class_indexes], dim=1)
        first_coef = (2. * first_intersection) / (first_predict_sum + first_target_sum)
        coef = first_coef

    if torch.sum(1 - target_class) > 0:
        zero_intersection = torch.sum((1 - predict[zero_class_indexes]) * (1 - target[zero_class_indexes]), dim=1)
        zero_predict_sum = torch.sum(1 - predict[zero_class_indexes], dim=1)
        zero_target_sum = torch.sum(1 - target[zero_class_indexes], dim=1)
        zero_coef = (2. * zero_intersection) / (zero_predict_sum + zero_target_sum)
        if torch.sum(target_class) > 0:
            coef = torch.cat((first_coef, zero_coef))
        else:
            coef = zero_coef

    if reduction == 'mean':
        coef = torch.mean(coef)
    return coef


class DiceLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predict, target):
        loss = 1 - dice_coeficient(predict=predict, target=target, reduction='None')
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class BCELoss(torch.nn.Module):
    def __init__(self, weights=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weights = weights

    def forward(self, predict, target, weight=(1, 1)):
        predict = predict.view(target.size(0), -1)
        target = target.view(target.size(0), -1)

        assert predict.shape[-1] == 1, "Data has more than 2 classes"

        pos_loss = weight[1] * target * torch.log(predict + 1e-10)
        neg_loss = weight[0] * (1.0 - target) * torch.log(1.0 - predict + 1e-10)
        if self.weights is not None:
            loss = self.weights[1] * pos_loss + self.weights[0] * neg_loss
        else:
            loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return torch.neg(torch.mean(loss))
        else:
            return torch.neg(torch.sum(loss))


# binary
class BFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2., weights=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.weights = weights  # weights is alpha

    def forward(self, predict, target, weight=(1, 1)):
        predict = predict.view(target.size(0), -1)
        target = target.view(target.size(0), -1)

        assert predict.shape[-1] == 1, "Data has more than 2 classes"

        pos_loss = weight[1] * target * (1.0 - predict) ** self.gamma * torch.log(predict+1e-10)
        neg_loss = weight[0] * (1.0 - target) * predict ** self.gamma * torch.log(1.0 - predict + 1e-10)

        if self.weights is not None:
            loss = self.weights[1] * pos_loss + self.weights[0] * neg_loss
        else:
            loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return torch.neg(torch.mean(loss))
        else:
            return torch.neg(torch.sum(loss))


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2., weights=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weights = weights
        self.gamma = gamma

    def forward(self, predict, target):
        predict = predict.view(target.size(0), predict.size(1), -1)  # (B, C, F)
        predict = predict.transpose(1, 2)  # (B, F, C)
        predict = predict.contiguous().view(-1, predict.size(2))  # (B*F, C)
        target = target.view(-1, 1)

        log_pt = F.log_softmax(predict, 1)
        ce = F.nll_loss(log_pt, target.squeeze(-1), self.weights, reduction=self.reduction)

        log_pt = log_pt.gather(-1, target)
        pt = Variable(log_pt.data.exp())
        focal_term = (1 - pt)**self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


# binary
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=2., beta=2., reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta

    def forward(self, predict, target):
        predict = predict.view(target.size(0), -1)
        target = target.view(target.size(0), -1)

        tp = torch.sum(predict * target, dim=1)
        fp = torch.sum(predict, dim=1) - tp
        fn = torch.sum(target, dim=1) - tp

        loss = 1 - (tp + 1e-10) / (tp + self.alpha*fp + self.beta*fn + 1e-10)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'None':
            return loss
        else:
            return torch.sum(loss)


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=2., beta=2., gamma=2., reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, reduction='None')

    def forward(self, predict, target):
        predict = predict.view(target.size(0), -1)
        target = target.view(target.size(0), -1)

        loss = self.tversky.forward(predict, target)
        loss = torch.pow(loss, self.gamma)

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)
