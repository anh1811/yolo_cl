"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #    print("__________________________________")
        #    print(self.lambda_box * box_loss)
        #    print(self.lambda_obj * object_loss)
        #    print(self.lambda_noobj * no_object_loss)
        #    print(self.lambda_class * class_loss)
        #    print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

def rpn_loss(pred_objectness_logits, pred_anchor_deltas, prev_pred_objectness_logits, prev_pred_anchor_deltas):
    loss = logit_distillation(pred_objectness_logits[0], prev_pred_objectness_logits[0])
    loss += anchor_delta_distillation(pred_anchor_deltas[0], prev_pred_anchor_deltas[0])
    return {"loss_dist_rpn": loss}


def backbone_loss(features, prev_features):
    loss = feature_distillation(features['res4'], prev_features['res4'])
    return {"loss_dist_backbone": loss}


def roi_head_loss(pred_class_logits, pred_proposal_deltas, prev_pred_class_logits, prev_pred_proposal_deltas, dist_loss_weight=0.5):
    loss = logit_distillation(pred_class_logits, prev_pred_class_logits)
    # loss = feature_distillation(pred_class_logits, prev_pred_class_logits)
    loss += anchor_delta_distillation(pred_proposal_deltas, prev_pred_proposal_deltas)
    return {"loss_dist_roi_head": dist_loss_weight * loss}


def logit_distillation(current_logits, prev_logits, T=6.0):
    p = F.log_softmax(current_logits / T, dim=1)
    q = F.softmax(prev_logits / T, dim=1)
    kl_div = torch.sum(F.kl_div(p, q, reduction='none').clamp(min=0.0) * (T**2)) / current_logits.shape[0]
    return kl_div


def anchor_delta_distillation(current_delta, prev_delta):
    # return smooth_l1_loss(current_delta, prev_delta, beta=0.1, reduction='mean')
    return F.mse_loss(current_delta, prev_delta)


def feature_distillation(features, prev_features):
    # return smooth_l1_loss(features, prev_features, beta=0.1, reduction='mean')
    return F.mse_loss(features, prev_features)
    # criterion_kd = Attention()
    # # criterion_kd = NSTLoss()
    # # criterion_kd = DistillKL(T=4)
    # # criterion_kd = FactorTransfer()
    # loss = criterion_kd(features, prev_features)
    # loss = torch.stack(loss, dim=0).sum()
    # return loss
