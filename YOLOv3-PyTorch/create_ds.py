import os
import numpy as np
import pandas as pd
import config


label_dir = 'PASCAL_VOC/labels'
train_dir = 'csv_path/2007_train.csv'
test_dir = 'csv_path/2007_test.csv'

ds_train = pd.read_csv(train_dir)
ds_test = pd.read_csv(test_dir)

# ds_train_new = ds_train[ds_train.iloc[:, 0].str.startswith('00')]
# ds_test_new = ds_test[ds_test.iloc[:, 0].str.startswith('00')]

# ds_train_new.to_csv("csv_path/2007_train.csv", index = False)
# ds_test_new.to_csv("csv_path/2007_test.csv", index =False)
only_out_label = []
path_with_newlb = []
base_classes = [i for i in range(config.BASE_CLASS)]
new_classes = [i for i in range(config.BASE_CLASS, config.NUM_CLASSES)]
assert len(base_classes) + len(new_classes) == config.NUM_CLASSES

for f in os.listdir(label_dir):
    label_path = os.path.join(label_dir, f)
    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
    # if len(bboxes) == 1 and int(bboxes[0][-1]) == 19:
    #     only_out_label.append(f)
    # else:
    path_contain_label = False
    is_only_new = True
    for i, bbox in enumerate(bboxes):
        if int(bbox[-1]) in new_classes:
            # path_with_newlb.append(f)
            path_contain_label = True
        if int(bbox[-1]) in base_classes:
            is_only_new = False
        if i == len(bboxes) - 1 and is_only_new:
            only_out_label.append(f)
    if path_contain_label:
        path_with_newlb.append(f)
print(len(only_out_label))
print(len(path_with_newlb))
ds_filter_train_base = ds_train.loc[~ds_train.iloc[:,1].isin(only_out_label)]
ds_filter_train_new = ds_train.loc[ds_train.iloc[:,1].isin(path_with_newlb)]
ds_filter_test_base = ds_test.loc[~ds_test.iloc[:,1].isin(only_out_label)]
ds_filter_test_new = ds_test.loc[ds_test.iloc[:,1].isin(path_with_newlb)]
ds_filter_train_base.to_csv(f"csv_path/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_train_base.csv", index = False)
ds_filter_train_new.to_csv(f"csv_path/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_train_new.csv", index = False)
ds_filter_test_base.to_csv(f"csv_path/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_test_base.csv", index = False)
ds_filter_test_new.to_csv(f"csv_path/2007_{config.BASE_CLASS}_{config.NEW_CLASS}_test_new.csv", index = False)