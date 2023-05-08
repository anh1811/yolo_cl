"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss, feature_distillation, roi_head_loss

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    is_base = config.BASE
    distill_ft = config.DISTILL_FEATURES
    distill_logit = config.DISTILL_LOGITS
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            features, out = model(x)
            if not is_base:
                prev_features, out = model.base_model(x)
                loss_distill = 0
                if distill_ft:
                    loss_distill += feature_distillation(prev_features, features)
                if distill_logit:
                    logits = out[..., 6: config.BASE_CLASS]
                    prev_logits = out[..., 6: config.BASE_CLASS]
                    anchors = out[..., :5]
                    prev_anchors = out[..., :5]
                    loss_distill += roi_head_loss(logits, anchors, prev_logits, prev_anchors)
                loss = (
                loss_distill + 
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
                )
            else:
                loss = ( 
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
                )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    if config.BASE:
        model = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
    else:
        model = YOLOv3(num_classes=config.BASE_CLASS + config.NEW_CLASS).to(config.DEVICE)
    distill_enable = config.DISTILL
    if distill_enable:
        base = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
        
        for param in base.parameters():
                param.requires_grad = False
        base.load_base_checkpoint(config.BASE_CHECK_POINT)
        '''load base model need to build later because dont know what to save in base model  '''
        model.base_model = base
    
    
    # print(model)
    # model.layers[15].pred[1] = CNNBlock(1024, 25 * 3, bn_act=True, kernel_size=1)
    # model.layers[22].pred[1] = CNNBlock(512, 25 * 3, bn_act=True, kernel_size=1)
    # model.layers[29].pred[1] = CNNBlock(256, 25 * 3, bn_act=True, kernel_size=1)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    # reload test and train 
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #check_class_accuracy(model, train_eval_loader, threshold=config.CONF_THRESHOLD)
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch % 10 == 0 and epoch > 0:
            print("On Test loader:")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")



if __name__ == "__main__":
    main()
