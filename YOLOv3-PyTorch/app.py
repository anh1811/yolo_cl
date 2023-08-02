# import streamlit as st
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
# import wandb
from model import YOLOv3
import cv2

IMAGE_SIZE = 416
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

infer_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ]
)

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

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

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

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
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
        w1 = boxes_preds[..., 2:3]
        h1 = boxes_preds[..., 3:4]
        w2 = boxes_labels[..., 2:3]
        h2 = boxes_labels[..., 3:4]
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
    iou = intersection / (box1_area + box2_area - intersection)
    if CIoU or DIoU or GIoU:
        cw = box1_x2.maximum(box2_x2) - box1_x1.minimum(box2_x1)  # convex (smallest enclosing box) width
        ch = box1_y2.maximum(box2_y2) - box1_y1.minimum(box2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((box2_x1 + box2_x2 - box1_x1 - box1_x2) ** 2 + (box2_y1 + box2_y2 - box1_y1 - box1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - intersection) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def resize_box(box, origin_dims, in_dims):
    # amount of padding
    h_ori, w_ori = origin_dims[0], origin_dims[1]
    print(h_ori, w_ori)
    padding_height = max(w_ori - h_ori, 0) * in_dims/w_ori
    padding_width =  max(h_ori - w_ori, 0) * in_dims/h_ori    
    
    #picture size after remove pad
    h_new = in_dims - padding_height
    w_new = in_dims - padding_width
    
    # resize box
    box[0] = (box[0] - padding_width//2)* w_ori/w_new
    box[1] = (box[1] - padding_height//2)* h_ori/h_new
    box[2] = (box[2] - padding_width//2)* w_ori/w_new
    box[3] = (box[3] - padding_height//2)* h_ori/h_new
    
    return box

def plot_image(image, boxes, im_ori):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im_ori = np.array(image)
    height, width, _ = im_ori.shape

    # # Create figure and axes
    # fig, ax = plt.subplots(1)
    # ax.set_axis_off()
    # # Display the image
    # ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        # box = resize_box(box, im_ori.shape[:2], 416)
        xmin = box[0] - box[2] / 2
        ymin = box[1] - box[3] / 2
        # print(box)
        im_ori = cv2.rectangle(im_ori, (int(xmin)* width, int(ymin)* height), (int(box[2]) * width, int(box[3])* height), colors[int(class_pred)], 3)
        cv2.imwrite('test_1.jpg',im_ori)
        label = class_labels[int(class_pred)]
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
        im_ori = cv2.putText(im_ori, label, (int(xmin) * width, int(ymin) * height + t_size[1] + 4), 
                              cv2.FONT_HERSHEY_PLAIN, 2, [225,255,255], 3)
        # rect = patches.Rectangle(
        #     (upper_left_x * width, upper_left_y * height),
        #     box[2] * width,
        #     box[3] * height,
        #     linewidth=2,
        #     edgecolor=colors[int(class_pred)],
        #     facecolor="none",
        # )
        # Add the patch to the Axes
        # ax.add_patch(rect)
    #     plt.text(
    #         upper_left_x * width,
    #         upper_left_y * height,
    #         s=class_labels[int(class_pred)],
    #         color="white",
    #         verticalalignment="top",
    #         bbox={"color": colors[int(class_pred)], "pad": 0},
    #     )
    # plt.savefig("test_9.png")
    # return fig2data(ax)
    return im_ori

def infer(model, img, thresh, iou_thresh, anchors):
    model.eval()
    image = np.array(img)
    # image = image[np.newaxis, :]
    augmentations = config.infer_transforms(image=image)
    x = augmentations["image"]
    # x = x.to("cuda")
    x = torch.reshape(x, [1,3,416,416])
    # print(x.shape)
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
        img = plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, img)
    return img

# scene = st.radio(
#     "Chọn bối cảnh",
#     ('19->20', '15->20', '10->20'))
scene = '19->20'

# task = st.radio(
#     "Chọn nhiệm vụ",
#     ('task1', 'task2', 'finetune'))

all = 20

if scene == '19->20':
    base = 19
    new = all - base
elif scene == '15->20':
    base = 15
    new = all - base
else:
    base = 10
    new = all - base

# if task == '1.Nhiệm vụ 1':
#     cls = base
#     task = 'task1'
# elif task == '2. Nhiệm vụ 2 (trước tinh chỉnh)':
#     cls = all
#     tune = False
# else:
#     cls = all
#     tune = True

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)


# uploaded_file = st.file_uploader("Chọn hình ảnh...", type=["jpg", "jpeg", "png"])
uploaded_file = '/home/ngocanh/Documents/final_thesis/code/dataset/19_1/base/images/test/000003.jpg'
image = Image.open(uploaded_file).convert("RGB")
print("Thuc hien bien doi")

#task 1
file_path = f"weights/2007_base_{base}_{new}_mAP_{base}_{new}.pth.tar"
model = YOLOv3(num_classes=base).to(device)
checkpoint = torch.load(file_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
image_1 = infer(model, image, 0.5, 0.5, scaled_anchors)

#task 2
file_path = f"weights/2007_task2_{base}_{new}_mAP_{base}_{new}.pth.tar"
model = YOLOv3(num_classes=all).to(device)
checkpoint = torch.load(file_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
image_2 = infer(model, image, 0.5, 0.5, scaled_anchors)

#ft
file_path = f"weights/2007_finetune_{base}_{new}_mAP_{base}_{new}.pth.tar"
checkpoint = torch.load(file_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
image_3 = infer(model, image, 0.5, 0.5, scaled_anchors)
# Streamlit App
# Widget tải lên file ảnh

# note = Image.open("note.png")
# st.image(note, width=150)


# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.image(image, caption="Ảnh đầu vào", use_column_width=True)
# with col2:
#     st.image(image_1, caption="Kết quả task 1", channels="BGR", use_column_width=True)
# with col3:
#     st.image(image_1, caption="Kết quả task 2 (no finetune)", channels="BGR", use_column_width=True)
# with col4:
#     st.image(image_1, caption="Kết quả task 2 (finetune)", channels="BGR", use_column_width=True)


import cv2
cv2.imwrite('test.jpg',image_1)

    # Hiển thị ảnh gốc

    # TODO: Đưa ảnh qua mô hình để xử lý (đoán, biến đổi, ...)

    # Hiển thị kết quả (ảnh sau khi qua mô hình), nếu có

    # Ví dụ: Nếu bạn đã có kết quả từ mô hình (processed_img) là một PIL Image
    # st.image(processed_img, caption="Processed Image", use_column_width=True)
