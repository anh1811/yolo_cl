"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
# import wandb
import warnings
warnings.filterwarnings("ignore")
import config
import torch
import torch.optim as optim
import os 
from store import Store
from model import YOLOv3, CNNBlock
from tqdm import tqdm
import mlflow
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
    load_base_checkpoint,
    get_image_store_load,
    preprocessing_image,
    Instances
)
from loss import YoloLoss, feature_distillation, loss_logits_dummy
from adan import Adan
torch.backends.cudnn.benchmark = True




# # wandb.init(project="my-project-name", entity="my-username")
# wandb.login(key="54aca131fa840c635c1b70e1e9aca363c47a21bd")
# # logger = WandbLogger(project="YOLOv3")
# wandb.init(project="my-project-name", entity="anhnn1811")

def train_fn(epoch, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, base_model, image_store):
    loop = tqdm(train_loader, leave=True)
    model.train_warp = config.TRAIN_WARP
    losses = []
    distill_losses = []
    yolo_losses = []
    is_base = config.BASE
    distill_ft = config.DISTILL_FEATURES
    distill_logit = config.DISTILL_LOGITS
    gamma = 0.2
    # if epoch >= 50:
    #     gamma = 0.5
    # print(image_store)
    for batch_idx, images in enumerate(loop):
        x = images["image"]
        y = images["label"]
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        # print(x.shape)
        # print(y0.shape)
        # print(torch.unique(y0[...,-1]))
        if (batch_idx + 1) % config.TRAIN_WARP_AT_ITR_NO == 0 and config.WARP:
            model.train_warp = True
            optimizer.zero_grad()
            x_store = image_store.retrieve()
            # print(len(x_store))
            if config.BASE:
                filter_cls = [i for i in range(config.BASE_CLASS)]
            else:
                filter_cls = [i for i in range(config.BASE_CLASS + config.NEW_CLASS)]
            # print(len(x_store))
            # load_small = get_image_store_load(x_store)
            # load_small = preprocess_image(x_store)
            if not config.USE_FEATURE_STORE:
                for i in range(0, len(x_store), config.BATCH_SIZE_WARP):
                    batched_images = x_store[i:i+config.BATCH_SIZE_WARP]
                    x_sample, y_sample = preprocessing_image(batched_images, filter_cls)
                    x_sample = x_sample.to(config.DEVICE)
                    y0_s, y1_s, y2_s = (
                    y_sample[0].to(config.DEVICE),
                    y_sample[1].to(config.DEVICE),
                    y_sample[2].to(config.DEVICE),
                    )
                    # print(torch.unique(y0_s[...,-1]))
                    with torch.cuda.amp.autocast():
                        out = model(x_sample)
                        # loss_distill = 0 
                        # if not is_base:
                        #     prev_out = base_model(x_sample)
                        #     features = model.get_features()
                        #     prev_features = model.get_features()
                        #     if distill_ft:
                        #         loss_distill += feature_distillation(prev_features, features)
                        #     if distill_logit:
                        #         loss_distill += loss_logits_dummy(out[0], prev_out[0], scaled_anchors[0])
                        #         + loss_logits_dummy(out[1], prev_out[1], scaled_anchors[1])
                        #         + loss_logits_dummy(out[2], prev_out[2], scaled_anchors[2])
                        
                        
                        warp_loss = ( 
                            loss_fn(out[0], y0_s, scaled_anchors[0])
                            + loss_fn(out[1], y1_s, scaled_anchors[1])
                            + loss_fn(out[2], y2_s, scaled_anchors[2])
                                )
                        

                        optimizer.zero_grad()
                        scaler.scale(warp_loss).backward()
                        for name, param in model.named_parameters():
                            if name not in config.WARP_LAYERS and param.grad is not None:
                                param.grad.fill_(0)
                        scaler.step(optimizer)
                        scaler.update()
            else:
                pass
                # optimizer.zero_grad()
                
            model.train_warp = False
        


        with torch.cuda.amp.autocast():
            out = model(x)
            # print(out[0].shape)
#             print(y0.shape)
            if not is_base and config.DISTILL:
                prev_out = base_model(x)
                features = model.get_features()
                prev_features = base_model.get_features()
                loss_distill = 0
                if distill_ft:
                    loss_distill += feature_distillation(prev_features, features)
                if distill_logit:
                    loss_yolo = loss_fn(out[0], y0, scaled_anchors[0], prev_out[0], gamma = gamma) + loss_fn(out[1], y1, scaled_anchors[1], prev_out[1], gamma = gamma)\
                    + loss_fn(out[2], y2, scaled_anchors[2], prev_out[2], gamma = gamma)
                else:
                    loss_yolo = ( 
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
                )
                
                # print(f'distill: {loss_distill}')
                # print(f'yolo: {loss_yolo}')
                loss =  loss_distill + loss_yolo
                # loss = oss_yolo
            else:
                loss = ( 
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
                )

        if config.DISTILL:
            distill_losses.append(loss_distill.item())
            yolo_losses.append(loss_yolo.item())

        if config.FINETUNE_NUM_IMAGE_PER_STORE < 0:
            #not update of finetuning
            image_store = update_image_store(image_store, images)

        if config.WARP:
            # image_store = update_image_store(image_store, images)
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            for name, param in model.named_parameters():
                if name in config.WARP_LAYERS:
                        param.grad.fill_(0)
        
            scaler.step(optimizer)
            scaler.update()
        else:
            # image_store = update_image_store(image_store, images)
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
    # update progress bar
        mean_loss = sum(losses) / len(losses)
        if config.DISTILL:
            mean_loss_dt = sum(distill_losses)/ len(distill_losses)
            mean_loss_yolo = sum(yolo_losses)/ len(yolo_losses)
            loop.set_postfix(loss_dis= mean_loss_dt, loss_yolo = mean_loss_yolo, loss =mean_loss)
            
            # mlflow.log_metric("")
        else:
            loop.set_postfix(loss=mean_loss)
    mlflow.log_metric("train_loss", mean_loss, step=epoch)
        # return mean_loss
        # wandb.log({"train_loss": mean_loss})
        

# def eval_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
import random
import numpy as np 
def update_image_store(image_store, images):
    image_paths = images["img_path"]
    labels_paths = images["label_path"]
    for (img,label) in zip(image_paths, labels_paths):
        ins = Instances(img, label)
        bboxes = np.roll(np.loadtxt(fname=label, delimiter=" ", ndmin=2), 4, axis=1)
        gt_classes_all = bboxes[:, -1].tolist()
        if config.BASE:
            # print(image_paths)
            gt_classes = [i for i in range(config.BASE_CLASS)]
        else:
            gt_classes = [i for i in range(config.BASE_CLASS, config.NUM_CLASSES)]
        cls = int(gt_classes[random.randrange(0, len(gt_classes))])
        # print(cls)
        # print(ins.label_path)
        image_store.add((ins,), (cls,))
    return image_store


def main():
    seed_everything()
    image_store = None
    x_base_store = None
    base = None
    if config.BASE:
        exp = f'2007_base_{config.BASE_CLASS}_{config.NEW_CLASS}'
    elif config.FINETUNE_NUM_IMAGE_PER_STORE > 0:
        exp = f'2007_finetune_{config.BASE_CLASS}_{config.NEW_CLASS}'
    else:
        exp = f'2007_task2_{config.BASE_CLASS}_{config.NEW_CLASS}'
    existing_exp = mlflow.get_experiment_by_name(exp)
    if not existing_exp:
        mlflow.create_experiment(exp)
    experiment = mlflow.set_experiment(exp)
    experiment_id = experiment.experiment_id
    # avg_score = {"loss": 0, "mAP": 0, "AP_newclass": 0}

    # if config.WARP:
    file_path = os.path.join(config.IMAGE_STORE_LOC, f'2007_image_store_base_{config.BASE_CLASS}_{config.NEW_CLASS}.pth')
    if os.path.exists(file_path) and not config.BASE:
        print(f"Open Image Store {file_path}")
        with open(file_path, "rb") as f:
            image_store = torch.load(f)
            # print(image_store)
            # print(len(image_store.retrieve())
        x_base_store = image_store.retrieve()
        print(f"Length of this Image store {len(x_base_store)}")
    else:
        print("Create Image Store")
        image_store = Store(config.NUM_CLASSES, config.NUM_IMAGES_PER_CLASS)
        x_base_store = None
    
    if config.FINETUNE_NUM_IMAGE_PER_STORE > 0:
        file_path = os.path.join(config.IMAGE_STORE_LOC, f'2007_image_store_task2_{config.BASE_CLASS}_{config.NEW_CLASS}.pth')
        print("Finetuning")
        if os.path.exists(file_path):
            print(f"Open Image Store {file_path}")
            with open(file_path, "rb") as f:
                image_store = torch.load(f)
            x_store = image_store.retrieve()
            print(f"Length of this Image store {len(x_store)}")
                # print(image_store)
                # print(len(image_store.retrieve()))
        else:
            raise ValueError

    model = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
    optimizer = Adan(params = model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
       
    if config.DISTILL and not config.BASE:
        
        base = YOLOv3(num_classes=config.BASE_CLASS).to(config.DEVICE)
        load_base_checkpoint(config.BASE_CHECK_POINT, base)
        for param in base.parameters():
                param.requires_grad = False
        print("Load Base model")
        '''load base model need to build later because dont know what to save in base model  '''
        # model.base_model = base
    
    # reload test and train

    if not config.BASE:
        print("Load weight from previous task")
        load_checkpoint(
            config.BASE_CHECK_POINT, model, optimizer, config.LEARNING_RATE
        )
        model.adaptation(layer_id = 15, num_class = config.NUM_CLASSES, in_feature = 1024, old_class = config.BASE_CLASS)
        model.adaptation(layer_id = 22, num_class = config.NUM_CLASSES, in_feature = 512, old_class = config.BASE_CLASS)
        model.adaptation(layer_id = 29, num_class = config.NUM_CLASSES, in_feature = 256, old_class = config.BASE_CLASS) 
        model.to(config.DEVICE)
    else:
        print("Load pretrainweight from darknet")
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    if config.FINETUNE_NUM_IMAGE_PER_STORE < 0:
        if config.BASE:
            print("Start Task 1:")
            train_loader, test_loader, train_eval_loader = get_loaders(
                train_csv_path=config.DATASET + f"/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_train_base.csv", test_csv_path=config.DATASET +\
                 f"/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_test_base.csv", x_store = x_base_store, base = config.BASE
            )
        else:
            print("Start Task 2:")
            train_loader, test_loader, train_eval_loader = get_loaders(
                train_csv_path=config.DATASET + f"/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_train_new.csv", test_csv_path=config.DATASET + "/2007_test.csv",\
                x_store = x_base_store, base = config.BASE
            )
            # print("Fin")
    else:
        print("Load check point task 2")
        load_base_checkpoint(config.CHECKPOINT_FILE, model)
        print("Load dataset for finetune:")
        train_loader, test_loader  = get_image_store_load(x_store, test_csv_path=config.DATASET + "/test.csv"
        )
    
    
    
    if config.LOAD_MODEL:
        print("Load check point")
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)


    # pred_boxes, true_boxes = get_evaluation_bboxes(
    #             test_loader,
    #             model,
    #             iou_threshold=config.NMS_IOU_THRESH,
    #             anchors=config.ANCHORS,
    #             threshold=config.CONF_THRESHOLD,
    #         )
    # mapval, _ = mean_average_precision(
    #             pred_boxes,
    #             true_boxes,
    #             iou_threshold=config.MAP_IOU_THRESH,
    #             box_format="midpoint",
    #             num_classes=config.NUM_CLASSES,
    #         )
    # print(f"MAP: {mapval.item()}")
    # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    # print("On Train Eval loader:")
    # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    mAP_best = 0
    mAP_new_best = 0
    # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    # print(config.NUM_EPOCHS)
    # print(optimizer.state_dict())
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_artifacts("code")
        for epoch in range(config.NUM_EPOCHS):
            # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
            train_fn(epoch, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, base_model = base, image_store=
                    image_store)

            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, filename=f"weights/checkpoint.pth.tar")
            
            #test
            if epoch % 10 == 0 and epoch > 0:
                print("On Test loader:")
                test_cls, test_obj = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
                # check_class_accuracy(base, test_loader, threshold=config.CONF_THRESHOLD)
    #             wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj})
                mlflow.log_metric("test_cls_precision", test_cls, step=epoch)
                mlflow.log_metric("test_obj_precision", test_obj, step=epoch)

                pred_boxes, true_boxes = get_evaluation_bboxes(
                    test_loader,
                    model,
                    iou_threshold=config.NMS_IOU_THRESH,
                    anchors=config.ANCHORS,
                    threshold=config.CONF_THRESHOLD,
                )
                mapval, ap_all = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=config.MAP_IOU_THRESH,
                    box_format="midpoint",
                    num_classes=config.BASE_CLASS if config.BASE else config.NUM_CLASSES,
                )
                print(f"MAP: {mapval.item()}")
                print(ap_all)
                new_ap = list(ap_all[-config.NEW_CLASS:])
                learn_ap = sum(new_ap)/len(new_ap)

                mlflow.log_metric("MAP", mapval.item(), step=epoch)
                mlflow.log_metric("AP_NEW", learn_ap, step=epoch)
                # mlflow.log_metric("ap_15", ap_all[-5], step=epoch)
                # mlflow.log_metric("ap_16", ap_all[-4], step=epoch)
                # mlflow.log_metric("ap_17", ap_all[-3], step=epoch)
                # mlflow.log_metric("ap_18", ap_all[-2], step=epoch)
                # mlflow.log_metric("ap_19", ap_all[-1], step=epoch)
                # print(f"test_cls_precsion: {test_cls}, test_obj: {test_obj}")
                # wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj, "MAP": mapval.item()})

                if mapval > mAP_best:
                    save_checkpoint(model, optimizer, filename=f"weights/{exp}_mAP_{config.BASE_CLASS}_{config.NEW_CLASS}.pth.tar")
                    mAP_best = mapval
                    mlflow.log_artifact(f"weights/{exp}_mAP_{config.BASE_CLASS}_{config.NEW_CLASS}.pth.tar")
                if learn_ap > mAP_new_best:
                    save_checkpoint(model, optimizer, filename=f"weights/{exp}_AP{config.NEW_CLASS}.pth.tar")
                    mAP_new_best = learn_ap
                    mlflow.log_artifact(f"weights/{exp}_AP{config.NEW_CLASS}.pth.tar")

    print("On Test loader:")
    test_cls, test_obj = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    # check_class_accuracy(base, test_loader, threshold=config.CONF_THRESHOLD)
#             wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj})
    mlflow.log_metric("test_cls_precision", test_cls, step=epoch)
    mlflow.log_metric("test_obj_precision", test_obj, step=epoch)

    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )
    mapval, ap_all = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.BASE_CLASS if config.BASE else config.NUM_CLASSES,
    )
    print(f"MAP: {mapval.item()}")
    print(ap_all)
    new_ap = list(ap_all[-config.NEW_CLASS:])
    learn_ap = sum(new_ap)/len(new_ap)

    mlflow.log_metric("MAP", mapval.item(), step=epoch)
    mlflow.log_metric("AP_NEW", learn_ap, step=epoch)
    # mlflow.log_metric("ap_15", ap_all[-5], step=epoch)
    # mlflow.log_metric("ap_16", ap_all[-4], step=epoch)
    # mlflow.log_metric("ap_17", ap_all[-3], step=epoch)
    # mlflow.log_metric("ap_18", ap_all[-2], step=epoch)
    # mlflow.log_metric("ap_19", ap_all[-1], step=epoch)
    # print(f"test_cls_precsion: {test_cls}, test_obj: {test_obj}")
    # wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj, "MAP": mapval.item()})

    if mapval > mAP_best:
        save_checkpoint(model, optimizer, filename=f"weights/{exp}_mAP_{config.BASE_CLASS}_{config.NEW_CLASS}.pth.tar")
        mAP_best = mapval
        mlflow.log_artifact(f"weights/{exp}_mAP_{config.BASE_CLASS}_{config.NEW_CLASS}.pth.tar")
    if learn_ap > mAP_new_best:
        save_checkpoint(model, optimizer, filename=f"weights/{exp}_AP{config.NEW_CLASS}.pth.tar")
        mAP_new_best = learn_ap
        mlflow.log_artifact(f"weights/{exp}_AP{config.NEW_CLASS}.pth.tar")
    

    if config.BASE:
        file_path = os.path.join(config.IMAGE_STORE_LOC, f'2007_image_store_base_{config.BASE_CLASS}_{config.NEW_CLASS}.pth')
    else:
        file_path = os.path.join(config.IMAGE_STORE_LOC, f'2007_image_store_task2_{config.BASE_CLASS}_{config.NEW_CLASS}.pth')
    if image_store is not None:
        with open(file_path, "wb") as f:
            torch.save(image_store, f)
    mlflow.log_artifact(file_path)
if __name__ == "__main__":
    main()