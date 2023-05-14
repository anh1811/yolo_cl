"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import wandb
import config
import torch
import torch.optim as optim
import os 
from store import Store
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
    seed_everything,
    load_base_checkpoint
)
from loss import YoloLoss, feature_distillation, loss_logits_dummy

torch.backends.cudnn.benchmark = True

# # wandb.init(project="my-project-name", entity="my-username")
# wandb.login(key="54aca131fa840c635c1b70e1e9aca363c47a21bd")
# # logger = WandbLogger(project="YOLOv3")
# wandb.init(project="my-project-name", entity="anhnn1811")

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, base_model, image_store):
    loop = tqdm(train_loader, leave=True)
    # model.train_warp = config.TRAIN_WARP
    losses = []
    is_base = config.BASE
    distill_ft = config.DISTILL_FEATURES
    distill_logit = config.DISTILL_LOGITS
    for batch_idx, (x, y) in enumerate(loop):
        if (batch_idx + 1) % config.TRAIN_WARP_AT_ITR_NO == 0 and config.WARP:
            model.train_warp = True
            optimizer.zero_grad()
            x_store = image_store.retrieve()
            if not config.USE_FEATURE_STORE:
                pass
            else:
                pass
                optimizer.zero_grad()
                
            model.train_warp = False
        
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
#             print(y0.shape)
            if not is_base:
                prev_out = base_model(x)
                features = model.get_features()
                prev_features = model.get_features()
                loss_distill = 0
                if distill_ft:
                    loss_distill += feature_distillation(prev_features, features)
                if distill_logit:
                    loss_distill += loss_logits_dummy(out[0], prev_out[0], scaled_anchors[0])
                    + loss_logits_dummy(out[1], prev_out[1], scaled_anchors[1])
                    + loss_logits_dummy(out[2], prev_out[2], scaled_anchors[2])
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
        
        if config.WARP:
            image_store = update_image_store(image_store, x, y)

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            for layer in model.layers():
                if layer in config.WARP_LAYERS:
                    layer.grad.fill_(0)
        
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
        # return mean_loss
        wandb.log({"train_loss": mean_loss})

# def eval_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
import random
def update_image_store(image_store, x, y):
    batch_size = x.shape(0)
    for i in range(batch_size):
        img = x[i,:]
        labels = y[0][i]
        labels = labels[labels[...,0] == 1.].tolist()
        labels = list(map(labels, int))
        cls = labels[random.randrange(0, len(labels))]            
    image_store.add((img, ), (cls, ))
    # for image in images:
    #     gt_classes = image["instances"].gt_classes
    #     cls = gt_classes[random.randrange(0, len(gt_classes))]
    #     self.image_store.add((image,), (cls,))
    return image_store


def main():
    seed_everything()
    image_store = None
    if config.WARP:
        file_path = os.path.join(config.IMAGE_STORE_LOC)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                image_store = torch.load(f)
        else:
            image_store = Store(config.NUM_CLASSES, config.NUM_IMAGES_PER_CLASS)
    model = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
        
    if config.DISTILL:
        base = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
        load_base_checkpoint(config.BASE_CHECK_POINT, base)
        for param in base.parameters():
                param.requires_grad = False
        
        '''load base model need to build later because dont know what to save in base model  '''
        # model.base_model = base
    
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
    if not config.BASE:
        load_checkpoint(
            config.BASE_CHECK_POINT, model, optimizer, config.LEARNING_RATE
        )
        model.layers[15].pred[1] = CNNBlock(1024, 25 * 3, bn_act=False, kernel_size=1)
        model.layers[22].pred[1] = CNNBlock(512, 25 * 3, bn_act=False, kernel_size=1)
        model.layers[29].pred[1] = CNNBlock(256, 25 * 3, bn_act=False, kernel_size=1)
        model.to(config.DEVICE)
    elif config.LOAD_MODEL or config.BASE:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    
    # print("On Train Eval loader:")
    # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    for epoch in range(config.NUM_EPOCHS):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, base_model = base, image_store=
                 image_store)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
        
        #test
        if epoch % 10 == 0 and epoch > 0:
            print("On Test loader:")
            test_cls, _, test_obj = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
#             wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj})
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
            wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj, "MAP": mapval.item()})


if __name__ == "__main__":
    main()
