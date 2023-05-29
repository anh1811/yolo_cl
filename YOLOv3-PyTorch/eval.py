"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim
# import wandb
from model import YOLOv3, CNNBlock
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    load_base_checkpoint,
    seed_everything
)
from loss import YoloLoss, loss_logits_dummy, feature_distillation

        # wandb.log({"train_loss": mean_loss})

def main():
    seed_everything()
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    # model.eval
    load_base_checkpoint('weights/Base_ml_stochweight_0.1_lm0.1_AP19.pth.tar', model)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler() 
    # base = None
    # if config.DISTILL:
    #     base = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
    #     load_base_checkpoint(config.BASE_CHECK_POINT, base)
    #     for param in base.parameters():
    #         param.requires_grad = False
        # model.base_model = base
    
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train_new.csv", test_csv_path=config.DATASET + "/test.csv"
    )
    
    #neu nhu khong phai gia doan base

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

            # wandb.log({"MAP": mapval.item()})
    # model.adaptation(layer_id = 15, num_class = config.NUM_CLASSES, in_feature = 1024, old_class = config.BASE_CLASS)
    # model.adaptation(layer_id = 22, num_class = config.NUM_CLASSES, in_feature = 512, old_class = config.BASE_CLASS)
    # model.adaptation(layer_id = 29, num_class = config.NUM_CLASSES, in_feature = 256, old_class = config.BASE_CLASS) 
    # model.to(config.DEVICE)
    check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    
    model.eval()
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )
    mapval, average_all_class = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES
    )
    print(f"MAP: {mapval.item()}")
    print(average_all_class)


if __name__ == "__main__":
    main()
