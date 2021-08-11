# Mask classifier 
## AI/ML Tutorial 
Date :10 Aug 21
## Project Overview
In light of covid-19 pandemic, Masks and Personal Protection Equipment serves to be one of the best protection against the COVID-19 virus. Masks are one of the most commonly available PPE however, its effectiveness deteroriates if not worn properly. Therefore, by creating an CV algorithm that is able to detect whether mask are worn, worn properly and worn at all would help to serve the community better. Reason Yolov4 was chosen over otehr algorithms as a first cut is due to yolov4 consisitent higher average precision at real-time detection at higher FPS, thus making it the ideal choice. For more information on YoloV4 perfomance refer to https://blog.roboflow.com/pp-yolo-beats-yolov4-object-detection/
### Target Classes
* `with_mask `​: Mask worn properly
* `without_mask `​: Mask not worn
* `mask_weared_incorrect `​: Mask worn, but incorrectly

# Sypnopsis of the problem. 
* **Dataset**: https://www.kaggle.com/andrewmvd/face-mask-detection
* **Image detection and classification**: The algorithm will determine if a detected face is wearing mask, wearing mask incorrectly or not wearing mask at all. The evaluation criteria is Mean Average Precision (MAP).

## Overview of Submitted folder
.
├── CV_YoloV4.ipynb
├── Face mask detection.zip
│   └── Annotations
│   └── Images
├── Results 
│   └── with_mask.txt
│   └── without_mask.txt
│   └── mask_weared_incorrect.txt
│   └── MAP.jpg

# CV_YoloV4.ipnyb
CV_yolov4 notebook converts .xml file into YOLOv4 format. In addition, CV_yolov4 noteboook generates data into train/validation split. Laslty, the code also shifts the necessary file cloned from gitclone into the respective folders.

# CV_Yolo_EDA.ipynb
Exploratory data anlysis file on the original dataset shows that there is a class imbalance (See Figure 1.). since this is a tutorial, I have included 100 additional data for mask_worn_incorrect so as to reduce the overall class inbalance. In reality however, we would be introducing more data to reduce the overall class inbalance.

![Image of results](https://github.com/kmt112/probable-lamp/blob/master/EDA_class%20inbalance.png)

# Gitclone
There are three seperate Gitfolders that ive cloned to create the algorithm. 

## 1. https://github.com/kmt112/probable-lamp
Contains configuration file obj.data, obj.names and config.cfg file that is required. Change config.cfg backup folder location according to where you want to store the backup weights.
* `train`​: locate train.txt folder, this is the data that YOLOV4 is training on. rememeber that .txt file has to be based in YOLOv4 format.
* `valid `​: locate valid.txt folder, this is the data that YOLOV4 is validating on and subsequently giving the Mean Average Precision.
* `config.cfg `​: base configuration file of YOLOv4. However some data augmentation parameters were introduced when training the model. Source : https://arxiv.org/pdf/2004.10934.pdf
### Data Augementation (located in config.cfg)
  * `width/height `​: Changed the resolution size to 416, increasing the width and height of yolov4 improves the resolution.
  * `batches `​: when batches are divided by subdivision, this determines the number of images that will be processes in parallel.
  * `satuation = 1.5, Hue = 1.5 `​: Mo
  * `mosaic = 1 `​: Mosaic data augemenration combines 4 training images into one in certain ratios (instead of two in cutmix). This prevents over-reliance on any key features.
  * `blur = 1 `​: blur will be applied randomly in 50% of the time.
  * `jitter = 0.3`​: randomly changes size of image and its aspect ratio.

## 2. https://github.com/AlexeyAB/darknet
Contains all the makefile and all other relevant folders required to run, compile and save the results of the custom object detector. 

* `Makefile`​: Before compiling the makefile, i changed  `OPENCV = 1` and `GPU = 1`
* `darknet/chart.png`​: the chart consists of MAP and losses after every iteration. MAP and losses are obtianed through the validation dataset. A total of 6,000 iterations has been done and the best weights have been saved at 5,800 iterations.

## 3. https://github.com/Cartucho/OpenLabeling (run on terminal (e.g. command prompt/ubuntu)
In order to supplement more data for the class inbalances, ive added another data repo https://github.com/cabani/MaskedFace-Net. The data taken from the above repo would require you to physically add a bounding box label it accordingly.

## Running of darknet Yolov4.
Download pretrained weights from COCO dataset.
```sh
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```
Compile file by running the following cell
```sh
!make
```
execute darknet
```sh
!chmod +x ./darknet
```
Initial Training of model
```sh
!./darknet detector train data/obj.data /content/darknet/cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map 
```
Picking up from last saved
```sh
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /content/drive/MyDrive/yolov4-obj_2700.weights -dont_show -map
```
Validation of dataset
```sh
!./darknet detector valid data/obj.data cfg/yolov4-obj.cfg /content/drive/MyDrive/yolov4-obj_bestfin.weights -dont_show -map
```
## Results
![Image of results](https://github.com/kmt112/probable-lamp/blob/master/Final%20Chart%5B3985%5D.png)

## License
MIT
Darknet Yolov4
