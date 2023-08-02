# YOLOv3 in PyTorch continual learning
 Continual Learning on YOLOv3 using knowledge distillation, replay sampling and meta-learning.

## Installation

### Clone and install requirements
```bash
# $ git clone https://github.com/anh1811/yolo_cl.git
$ cd YOLOv3-PyTorch
$ pip install requirements.txt
```

### Dowload original weights 
Download YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights. Save the weights to PyTorch format by running the model_with_weights.py file.

### Download Pascal VOC dataset
Download the preprocessed dataset from [link](https://www.kaggle.com/aladdinpersson/pascal-voc-yolo-works-with-albumentations). Just unzip this in the main directory.
The file structure of the dataset is a folder with images, a folder with corresponding text files containing the bounding boxes and class targets for each image and two csv-files containing the subsets of the data used for training and testing. 


### Training
Edit the config.py file to match the setup you want to use. Then run train.py

<!-- ### Results
| Model                   | mAP @ 50 IoU |
| ----------------------- |:-----------------:|
| YOLOv3 (Pascal VOC) 	  | 78.2              |
| YOLOv3 (MS-COCO)        | Not done yet      |

The model was evaluated with confidence 0.2 and IOU threshold 0.45 using NMS. -->

<!-- ## YOLOv3 paper 
The implementation is based on the following paper:
### An Incremental Improvement 
by Joseph Redmon, Ali Farhadi -->



```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
