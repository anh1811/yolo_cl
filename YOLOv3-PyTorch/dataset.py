"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image,
    Instances
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
import random 



class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        instance = None,
        preprocessing = None,
        transform=None,
        filter_dataset = None,
        auxilary = None,
    ):
        self.annotations = pd.read_csv(csv_file)

        self.preprocessing = preprocessing
        # self.auxilary = auxilary
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.filter_dataset = filter_dataset
        self.instances = instance
        self.filter_last_task = [i for i in range(config.BASE_CLASS)]
        self.size_limit = 0.01
        # if auxilary is not None:
        #     self.auxilary = pd.read_csv(auxilary).sample(300)
        # else:

    def load_mosaic_image_and_boxes(self, main_image, main_boxes, main_index, s=416, 
                                    scale_range=[0.3, 0.7], maxfrac=0.75):
        self.mosaic_size = s
        scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        xc = int(scale_x * s)
        yc = int(scale_y * s)
 
        # random other 3 sample (could be the same too...)
        # list_wout_mindex = self.instances.copy()
        # list_wout_mindex.remove(main_index)
        indices = [main_index]
        while main_index in indices:
            indices = random.sample(range(len(self.instances)), 4)
            index_of_main_image = random.randint(0,3)


        mosaic_image = np.zeros((s, s, 3), dtype=np.float32)
        final_boxes  = []
        final_labels = []

        # random.shuffle(indices)
        for i, index in enumerate(indices):
            if i == 0:    # top left
                x1a, y1a, x2a, y2a =  0,  0, xc, yc
                delta_x = s - xc
                delta_y = s - yc
                height = yc 
                width = xc
                # x1b, y1b, x2b, y2b = s - xc, s - yc, s, s # from bottom right
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, 0, s , yc
                delta_x = s - xc
                delta_y = s - yc
                height = yc
                width= s - xc
                # x1b, y1b, x2b, y2b = 0, s - yc, s - xc, s # from bottom left
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = 0, yc, xc, s
                height = s - yc
                width= xc
                # x1b, y1b, x2b, y2b = s - xc, 0, s, s-yc   # from top right
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc,  s, s
                height = s - yc
                width= s - xc
            preprocessing = config.train_preprocess(height=height, width=width)
            #load instaces or images
            if i == index_of_main_image:
                augments = preprocessing(image = main_image, bboxes = main_boxes)
                image = augments["image"]
                boxes = augments["bboxes"]
            elif random.random() < 0.5:
                boxes = []
                while len(boxes) == 0:
                    image, boxes = self.load_instance(index, preprocessing=preprocessing)
            else:
                image, boxes, _, _ = self.load_bboxes_image(index)
                augments = preprocessing(image = image, bboxes = boxes)
                image = augments["image"]
                boxes = augments["bboxes"]
            mosaic_image[y1a:y2a, x1a:x2a] = image
            if len(boxes) == 0:
                continue
            boxes = np.asarray(boxes)
            boxes[:, [0,2]] *= width
            boxes[:, [1,3]] *= height

            boxes[:, 0] += x1a
            boxes[:, 1] += y1a

            boxes[:, [0,2]] /= s
            boxes[:, [1,3]] /= s
            # print(boxes)
            # print("this is boxes")
            # print(boxes)
            # new_boxes = np.zeros_like(boxes)
            # print(image.shape)
            # for bbox in img_annos:

            #x_c,y_c, w, h ->xyxy
            # new_boxes[:, 0] = boxes[:,0] - boxes[:,2]*0.5
            # new_boxes[:, 1] = boxes[:,1] - boxes[:,3]*0.5
            # new_boxes[:, 2] = boxes[:,0] + boxes[:,2]*0.5
            # new_boxes[:, 3] = boxes[:,1] + boxes[:,3]*0.5
            # print(f"new_box {new_boxes}")
            # del bbox
            

            # calculate and apply box offsets due to replacement            
            # offset_x = x1a - x1b
            # offset_y = y1a - y1b
            # print(offset_x/s, offset_y/s)
            # new_boxes[:, 0] += offset_x/s
            # new_boxes[:, 1] += offset_y/s
            # new_boxes[:, 2] += offset_x/s
            # new_boxes[:, 3] += offset_y/s
            # print(new_boxes)
            # new_boxes[:, :-1] = np.clip(new_boxes[:, :-1], 0.000001, 1.)
            
            # print(f"this is {new_boxes}")
            # boxes[:, 0] = (new_boxes[:,0] + new_boxes[:,2]) / 2.
            # boxes[:, 1] = (new_boxes[:,1] + new_boxes[:,3]) / 2.
            # boxes[:, 2] = new_boxes[:,2] - new_boxes[:, 0]
            # boxes[:, 3] = new_boxes[:,3] - new_boxes[:,1]
            # print(f"this is {boxes}")
            # print(y1a, y2a, x1a, x2a)
            # print(y1b, y2b, x1b, x2b)
            # cut image, save boxes
            # print(boxes)
            
            final_boxes.append(boxes)

        # collect boxes
        final_boxes  = np.vstack(final_boxes)
        # print(final_boxes)
        # clip boxes to the image area
        # final_boxes[:, :-1] = np.clip(final_boxes[:, :-1], 0.000001, 1.)
        
        # w = (final_boxes[:,2] - final_boxes[:,0])
        # h = (final_boxes[:,3] - final_boxes[:,1])
        
        # discard boxes where w or h <10
        # final_boxes = final_boxes[(final_boxes[:,2]>=self.size_limit) & (final_boxes[:,3]>=self.size_limit)]

        return mosaic_image, final_boxes
    
    def load_instance(self, index, preprocessing):
        label_path = self.instances[index].label_path.split('/')[-1]
        label_path = os.path.join(config.DATASET + '/labels', label_path)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        # print(label_path)
        # print(len(bboxes))
        # print(self.filter_dataset)
        if self.filter_last_task:
            bboxes = [box for box in bboxes if int(box[-1]) in self.filter_last_task]
            # print(len(bboxes))
        img_path = self.instances[index].img_path.split('/')[-1]
        img_path = os.path.join(config.DATASET + '/images', img_path)
        # print(bboxes)
        # bboxes = np.asarray(bboxes)
        image = np.array(Image.open(img_path).convert("RGB"))


        
        augmentations = preprocessing(image=image, bboxes=bboxes)
        image = augmentations["image"]
        boxes = augmentations["bboxes"]
        # if self.preprocessing:
        #     augmentations = self.preprocessing(image=image, bboxes=bboxes)
        #     image = augmentations["image"]
        #     boxes = augmentations["bboxes"]

        # # as pascal voc format
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # width  to xmax - by: xmin + xmax 
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # height to ymax - by: ymin + ymax
        
        # #resize images and boxes
        # if size != 1:
        #     f_y = size/image.shape[0]
        #     f_x = size/image.shape[1]
            
        #     image = cv2.resize(image, (size, size))
        
        #     boxes[:, 0] = boxes[:, 0]*f_x
        #     boxes[:, 2] = boxes[:, 2]*f_x
        #     boxes[:, 1] = boxes[:, 1]*f_y
        #     boxes[:, 3] = boxes[:, 3]*f_y
        
        return image, boxes

    def __len__(self):
        return len(self.annotations)

    def load_bboxes_image(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # print(label_path)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        if self.filter_dataset:
            bboxes = [box for box in bboxes if int(box[-1]) in self.filter_dataset]
            # print(len(bboxes))
        # bboxes = np.asarray(bboxes)
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        # print(img_path)
        # print(label_path)
        image = np.array(Image.open(img_path).convert("RGB"))
        return image, bboxes, label_path, img_path

    def __getitem__(self, index):
        
        input_ds = dict()
        # print(len(bboxes))
        # print(self.filter_dataset)
        
        image, bboxes, label_path, img_path = self.load_bboxes_image(index)
        if self.preprocessing:
            augmentations = self.preprocessing(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        # print(bboxes.type)
        # if random.random() < 0.5:
        if True:
            #load random instance from the image store
            image, bboxes = self.load_mosaic_image_and_boxes(image, bboxes, index)
            # print(bboxes)
            #perform mosaice on the image + instance 
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='trunc')
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
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

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        input_ds["img_path"] = img_path
        input_ds["image"] = image
        input_ds["label"] = tuple(targets)
        input_ds["label_path"] = label_path
        return input_ds


class ImageStore(Dataset):
    def __init__(
        self,
        instances,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
        filter_dataset = None,
    ):
        # self.annotations = pd.read_csv(csv_file)
        self.images = instances
        # self.label_dir = label_dir
        # self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.filter_dataset = filter_dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        input_ds = dict()
        label_path = self.images[index].label_path
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        # print(len(bboxes))
        # print(self.filter_dataset)
        if self.filter_dataset:
            bboxes = [box for box in bboxes if int(box[-1]) in self.filter_dataset]
            # print(len(bboxes))
        img_path = self.images[index].img_path
        
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='trunc')
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
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

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        
        input_ds["img_path"] = img_path
        input_ds["image"] = image
        input_ds["label"] = tuple(targets)
        input_ds["label_path"] = label_path
        
        return input_ds
# def collat_fn()

def test():
    anchors = config.ANCHORS
    
    transform = config.test_transforms
    IMAGE_SIZE = 32
    file_path = os.path.join('./weights', f'image_store_base_15_5.pth')

    print("Finetuning")
    if os.path.exists(file_path):
        print(f"Open Image Store {file_path}")
        with open(file_path, "rb") as f:
            image_store = torch.load(f)
        x_store = image_store.retrieve()
    
    dataset = YOLODataset(
        'csv_path/19_1_train_new.csv',
        instance= x_store,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        filter_dataset = [19],
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    for load in loader:
        x = load["image"]
        y = load["label"]
        # print(load["img_path"])
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(x.shape)
            # print(y[i].shape)
            # print(y[i][y[i][...,0] == 1.].shape)
            # print(y[i])
            # print(y[i][..., 0])
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(boxes)
        # break
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
        break




if __name__ == "__main__":
    test()