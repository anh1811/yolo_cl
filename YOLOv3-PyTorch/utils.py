import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL.Image as Image


class Instances():
    def __init__(
        self,
        img_path,
        label_path
    ):
        self.img_path = img_path
        self.label_path = label_path 
        
def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1


        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions), average_precisions


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.savefig("test.png")


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, images in enumerate(tqdm(loader)):
        x = images["image"]
        labels = images["label"]
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, images in enumerate(tqdm(loader)):
        # if idx == 100:
        #     break
        x = images["image"]
        y = images["label"]
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)
            test_cls = (correct_class/(tot_class_preds+1e-16))*100
            test_nobj = (correct_noobj/(tot_noobj+1e-16))*100
            test_obj = (correct_obj/(tot_obj+1e-16))*100
    
    print(f"Class accuracy is: {test_cls:2f}%")
    print(f"No obj accuracy is: {test_nobj:2f}%")
    print(f"Obj accuracy is: {test_obj:2f}%")
    # wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj})
    model.train()
    return test_cls, test_obj


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_base_checkpoint(checkpoint_file, model):
    '''
        load_checkpoint model and set require_grad = False
    '''
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path, x_store, base = False):
    from dataset import YOLODataset
    classes_all = [i for i in range(config.NUM_CLASSES)]
    if base:
        classes = classes_all[:config.BASE_CLASS]
    else:
        # classes = [i for i in range(config.BASE_CLASS + config.NEW_CLASS)]
        classes = classes_all[-config.NEW_CLASS:]

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        instance = x_store,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        train = True,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        filter_dataset = classes
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        filter_dataset = classes if base else classes_all
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader


def get_image_store_load(x_store, test_csv_path):
    from dataset import ImageStore, YOLODataset
    if config.BASE:
        classes = [i for i in range(config.BASE_CLASS)]
    else:
        classes = [i for i in range(config.BASE_CLASS + config.NEW_CLASS)]
    IMAGE_SIZE = config.IMAGE_SIZE
    img_store_dataset = ImageStore(
        instances=x_store,
        anchors=config.ANCHORS,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        filter_dataset = classes
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        filter_dataset = classes
    )
    loader = DataLoader(
        dataset=img_store_dataset,
        batch_size=config.BATCH_SIZE_FINETUNE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    return loader, test_loader

import numpy as np 

def preprocessing_image(x_store, filter_dataset):
    import config
    from PIL import Image, ImageFile
    anchors = config.ANCHORS
    anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
    num_anchors = anchors.shape[0]
    num_anchors_per_scale = num_anchors // 3
    IMAGE_SIZE = config.IMAGE_SIZE
    S_anchor = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    ignore_iou_thresh = 0.5
    x_list = list()
    y_0_list = list()
    y_1_list = list()
    y_2_list = list()
    for img in x_store:
        img_path = img.img_path
        label_path = img.label_path
        image = np.array(Image.open(img_path).convert("RGB"))
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        if filter_dataset:
            bboxes = [box for box in bboxes if int(box[-1]) in filter_dataset]
        train_preprocess = config.train_preprocess()
        preprocessing = train_preprocess(image=image, bboxes=bboxes)
        augmentations = config.train_transforms(image=preprocessing["image"], bboxes=preprocessing["bboxes"])
        image = augmentations["image"]
        bboxes = augmentations["bboxes"]
        targets = [torch.zeros((num_anchors // 3, S, S, 6)) for S in S_anchor]
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = torch.div(anchor_idx, num_anchors_per_scale, rounding_mode='trunc')
                anchor_on_scale = anchor_idx % num_anchors_per_scale
                S = S_anchor[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction        
        x_list.append(image.to(config.DEVICE))
        # y_list.append(tuple(targets))
        y_0_list.append(targets[0].to(config.DEVICE))
        y_1_list.append(targets[1].to(config.DEVICE))
        y_2_list.append(targets[2].to(config.DEVICE))
    y_0_list = torch.stack(y_0_list)
    y_1_list = torch.stack(y_1_list)
    y_2_list = torch.stack(y_2_list)
    x_list = torch.stack(x_list)
    # print(y_0_list.shape)
    y_list = [y_0_list, y_1_list, y_2_list]
    return x_list, y_list    

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

def infer(model, img_path, thresh, iou_thresh, anchors):
    model.eval()
    image = np.array(Image.open(img_path).convert("RGB"))
    # image = image[np.newaxis, :]
    augmentations = config.infer_transforms(image=image)
    x = augmentations["image"]
    x = x.to("cuda")
    x = torch.reshape(x, [1,3,416,416])
    print(x.shape)
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    
    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
Overwriting utils.py
%%writefile train.py
"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import wandb
import warnings
warnings.filterwarnings("ignore")
import config
import torch
import torch.optim as optim
import os 
from store import Store
from model import YOLOv3, CNNBlock
from tqdm import tqdm
# import mlflow
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




def train_fn(epoch, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, base_model, image_store, meta_optimizer):
    
#     params = []
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             params.append(name)
#     print(len(params))
    loop = tqdm(train_loader, leave=True)
    model.train_warp = config.TRAIN_WARP
    losses = []
    distill_losses = []
    yolo_losses = []
    is_base = config.BASE
    distill_ft = config.DISTILL_FEATURES
    distill_logit = config.DISTILL_LOGITS
    gamma = 0.5
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
#         with torch.cuda.amp.autocast():
#             rand = torch.rand(1, 3, 416, 416).to(config.DEVICE)
# #             print(x.shape)
#             out_prev = model(rand)
        if (batch_idx + 1) % config.TRAIN_WARP_AT_ITR_NO == 0 and config.WARP:
#             scaler_state = scaler.state_dict()
            model.train_warp = True
            optimizer.zero_grad()
            x_store = image_store.retrieve()
            # print(len(x_store))
            if config.BASE:
                filter_cls = [i for i in range(config.BASE_CLASS)]
            else:
                filter_cls = [i for i in range(config.BASE_CLASS)]
            # print(len(x_store))
            # load_small = get_image_store_load(x_store)
            # load_small = preprocess_image(x_store)
#             params = []
            if not config.USE_FEATURE_STORE:
                first = True
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
                        

                        meta_optimizer.zero_grad()
                        scaler.scale(warp_loss).backward()
#                         if first:
#                             new_param = [name for (name, param) in model.named_parameters() if param.grad is not None]
#                             first = False
#                             print(set(params).difference(set(new_param)))
#                             print(len(set(new_param)))
#                             print(len(set(params)))
                        

                        for name, param in model.named_parameters():
                            if name not in config.WARP_LAYERS and param.grad is not None:
#                              if param.grad is not None:
                                param.grad.zero_()

                        scaler.step(meta_optimizer)
                        scaler.update()
            

                
            else:
                pass
                # optimizer.zero_grad()
#             scaler.load_state_dict(scaler_state)
            model.train_warp = False
#             del out
        


        with torch.cuda.amp.autocast():
#             out = model(rand)
#             print(torch.equal(out[0], out_prev[0]))
            out = model(x)
#             for (p1, p2) in zip(model1.parameters(), model2.parameters()):
#                 if p1.data.ne(p2.data).sum() > 0:
#                     return False
#                 return True
#             print(torch.equal(out, out_prev))
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
                loss =  gamma * loss_distill + loss_yolo
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

        if config.WARP:
            image_store = update_image_store(image_store, images)
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# scaler.step(optimizer)
# scaler.update()
            for name, param in model.named_parameters():
                if name in config.WARP_LAYERS:
                        param.grad.zero_()
        
            scaler.step(optimizer)
            scaler.update()
        else:
            image_store = update_image_store(image_store, images)
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
#     mlflow.log_metric("train_loss", mean_loss, step=epoch)
        # return mean_loss
        # wandb.log({"train_loss": mean_loss})
        wandb.log({"train_loss": mean_loss})

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
#     existing_exp = mlflow.get_experiment_by_name(exp)
    
    # wandb.init(project="my-project-name", entity="my-username")
    wandb.login(key="54aca131fa840c635c1b70e1e9aca363c47a21bd")
    # logger = WandbLogger(project="YOLOv3")
    wandb.init(project=f"{exp}", entity="anhnn1811")
#     if not existing_exp:
#         mlflow.create_experiment(exp)
#     experiment = mlflow.set_experiment(exp)
#     experiment_id = experiment.experiment_id
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
        file_path = os.path.join(config.IMAGE_STORE_LOC, f'image_store_task2_{config.BASE_CLASS}_{config.NEW_CLASS}.pth')
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
    optimizer = optim.AdamW(params = model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    meta_optimizer = optim.SGD(params = model.parameters(), lr = config.LEARNING_RATE)
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
                train_csv_path=f"csv_path/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_train_base.csv", test_csv_path=f"csv_path/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_test_base.csv", x_store = x_base_store, base = config.BASE
            )
        else:
            print("Start Task 2:")
            train_loader, test_loader, train_eval_loader = get_loaders(
                train_csv_path= f"csv_path/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_train_new.csv", test_csv_path="csv_path/2007_test.csv",\
                x_store = x_base_store, base = config.BASE
            )
            # print("Fin")
    else:
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
    for epoch in range(config.NUM_EPOCHS):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(epoch, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, base_model = base, image_store=
                image_store, meta_optimizer = meta_optimizer)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #test
        if epoch % 10 == 0 and epoch > 0:
            print("On Test loader:")
            test_cls, test_obj = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            # check_class_accuracy(base, test_loader, threshold=config.CONF_THRESHOLD)
            wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj})
#                 mlflow.log_metric("test_cls_precision", test_cls, step=epoch)
#                 mlflow.log_metric("test_obj_precision", test_obj, step=epoch)

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

#                 mlflow.log_metric("MAP", mapval.item(), step=epoch)
#                 mlflow.log_metric("AP_NEW", learn_ap, step=epoch)
            # mlflow.log_metric("ap_15", ap_all[-5], step=epoch)
            # mlflow.log_metric("ap_16", ap_all[-4], step=epoch)
            # mlflow.log_metric("ap_17", ap_all[-3], step=epoch)
            # mlflow.log_metric("ap_18", ap_all[-2], step=epoch)
            # mlflow.log_metric("ap_19", ap_all[-1], step=epoch)
            # print(f"test_cls_precsion: {test_cls}, test_obj: {test_obj}")
            wandb.log({"MAP": mapval.item()})
            wandb.log({"NEW_AP": learn_ap})

            if mapval > mAP_best:
                save_checkpoint(model, optimizer, filename=f"{exp}_mAP_{config.BASE_CLASS}_{config.NEW_CLASS}.pth.tar")
                mAP_best = mapval
#                     mlflow.log_artifact(f"weights/{exp}_mAP_{config.BASE_CLASS}_{config.NEW_CLASS}.pth.tar")
            if learn_ap > mAP_new_best:
                save_checkpoint(model, optimizer, filename=f"{exp}_AP{config.NEW_CLASS}.pth.tar")
                mAP_new_best = learn_ap
#                     mlflow.log_artifact(f"weights/{exp}_AP{config.NEW_CLASS}.pth.tar")
    print("On Test loader:")
    test_cls, test_obj = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    # check_class_accuracy(base, test_loader, threshold=config.CONF_THRESHOLD)
    wandb.log({"test_cls_precision": test_cls, "test_obj_precsion": test_obj})
#                 mlflow.log_metric("test_cls_precision", test_cls, step=epoch)
#                 mlflow.log_metric("test_obj_precision", test_obj, step=epoch)

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

#                 mlflow.log_metric("MAP", mapval.item(), step=epoch)
#                 mlflow.log_metric("AP_NEW", learn_ap, step=epoch)
    # mlflow.log_metric("ap_15", ap_all[-5], step=epoch)
    # mlflow.log_metric("ap_16", ap_all[-4], step=epoch)
    # mlflow.log_metric("ap_17", ap_all[-3], step=epoch)
    # mlflow.log_metric("ap_18", ap_all[-2], step=epoch)
    # mlflow.log_metric("ap_19", ap_all[-1], step=epoch)
    # print(f"test_cls_precsion: {test_cls}, test_obj: {test_obj}")
    wandb.log({"MAP": mapval.item()})
    wandb.log({"NEW_AP": learn_ap})
    if config.BASE:
        file_path = os.path.join(f'2007_image_store_base_{config.BASE_CLASS}_{config.NEW_CLASS}.pth')
    else:
        file_path = os.path.join(f'2007_image_store_task2_{config.BASE_CLASS}_{config.NEW_CLASS}.pth')
    if image_store is not None:
        with open(file_path, "wb") as f:
            torch.save(image_store, f)
#     mlflow.log_artifact(file_path)
if __name__ == "__main__":
    main()