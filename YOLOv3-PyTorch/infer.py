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
    seed_everything,
    infer
)
from loss import YoloLoss, loss_logits_dummy, feature_distillation

        # wandb.log({"train_loss": mean_loss})

def main():
    seed_everything()
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.eval()
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    load_base_checkpoint('weights/2007_finetune_19_1_mAP_19_1.pth.tar', model)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler() 
    # base = None
    # if config.DISTILL:
    #     base = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
    #     load_base_checkpoint(config.BASE_CHECK_POINT, base)
    #     for param in base.parameters():
    #         param.requires_grad = False
        # model.base_model = base
    
    img_path = '/home/ngocanh/Documents/final_thesis/code/dataset/PASCAL_VOC/images/007949.jpg'
    infer(model, img_path, 0.7, 0.5, scaled_anchors)

if __name__ == "__main__":
    main()