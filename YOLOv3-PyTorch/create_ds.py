import os
import numpy as np
import pandas as pd

label_dir = '/home/ngocanh/Documents/final_thesis/code/yolo_cl/YOLOv3-PyTorch/PASCAL_VOC/labels'
train_dir = '/home/ngocanh/Documents/final_thesis/code/yolo_cl/YOLOv3-PyTorch/PASCAL_VOC/train.csv'
test_dir = '/home/ngocanh/Documents/final_thesis/code/yolo_cl/YOLOv3-PyTorch/PASCAL_VOC/test.csv'

ds_train = pd.read_csv(train_dir)
ds_test = pd.read_csv(test_dir)
only_out_label = []
path_with_newlb = []
for f in os.listdir(label_dir):
    label_path = os.path.join(label_dir, f)
    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
    if len(bboxes) == 1 and bboxes[0][-1] == 19:
        only_out_label.append(f)
    else:
        for bbox in bboxes:
            if bbox[-1] == 19:
                path_with_newlb.append(f)
                break
print(len(only_out_label))
print(len(path_with_newlb))
ds_filter_train_base = ds_train.loc[~ds_train.iloc[:,1].isin(only_out_label)]
ds_filter_train_new = ds_train.loc[ds_train.iloc[:,1].isin(path_with_newlb)]
ds_filter_test_base = ds_test.loc[~ds_test.iloc[:,1].isin(only_out_label)]
ds_filter_test_new = ds_test.loc[ds_test.iloc[:,1].isin(path_with_newlb)]
ds_filter_train_base.to_csv("train_base.csv", index = False)
ds_filter_train_new.to_csv("train_new.csv", index = False)
ds_filter_test_base.to_csv("test_base.csv", index = False)
ds_filter_test_new.to_csv("test_new.csv", index = False)