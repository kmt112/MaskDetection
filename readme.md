# Mask classifier 
## AI/ML Tutorial 
Date :10 Aug 21
## Project Overview
In light of covid-19 pandemic, Masks and Personal Protection Equipment (PPE) serves to be one of the best protection against the COVID-19 virus. Masks are one of the most commonly available PPE however, its effectiveness deteroriates if not worn properly. Therefore, by creating an CV algorithm that is able to detect whether mask are worn, worn incorrectly or not worn at all would help to serve the community better. The model has to react fast therefore requiring lesser computational power than computationally heavy model. this makes yolov4 an ideal algorithm. Yolov4 has consisitently higher average precision at real-time detection(FPS: > 45) in addition yolov4 is a one-stage object detector thus making it computationally lighter. For more information on YoloV4 perfomance refer to https://blog.roboflow.com/pp-yolo-beats-yolov4-object-detection/
### Target Classes
* `with_mask `​: Mask worn properly
* `without_mask `​: Mask not worn
* `mask_weared_incorrect `​: Mask worn, but incorrectly

# Sypnopsis of the problem. 
* **Dataset**: https://www.kaggle.com/andrewmvd/face-mask-detection
* **Image detection and classification**: The algorithm will determine if a detected face is wearing mask, wearing mask incorrectly or not wearing mask at all. The evaluation criteria is Mean Average Precision (MAP).

# CV_Yolo_EDA.ipynb
Exploratory data anlysis file on the original dataset shows that there is a class imbalance (See Figure 1.). since this is a tutorial, I have included 100 additional data for mask_worn_incorrect so as to reduce the overall class inbalance. In reality however, we would be introducing more data to reduce the overall class inbalance.

![Image of results](https://github.com/kmt112/probable-lamp/blob/master/EDA_class%20inbalance.png)

# CV_YoloV4_final.ipnyb
CV_yolov4 notebook converts .xml file into YOLOv4 format. In addition, CV_yolov4 noteboook generates data into train/validation split. The code also shifts the necessary file cloned from gitclone into the respective folders. Finally, darknet is being compiled and run. Weight are generated over 6000 iterations and the best weights are saved in google drive.

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
Download pretrained weights from COCO dataset for transfer learning. This is only used in the first iteration, thereafter you will used the pretrained weights that your model have saved.
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
!./darknet detector valid data/obj.data cfg/yolov4-obj.cfg /content/drive/MyDrive/yolov4-obj_oldbest.weights -dont_show -map
```
# Results
<img src="https://github.com/kmt112/probable-lamp/blob/master/results/Final%20Chart%202.jpg" width="800" height="500"> <img src="https://github.com/kmt112/probable-lamp/blob/master/results/Final%20Chart.png" width="800" height="500">


The initial model(top) was trained with the default data, the latter model was retrained with new data inclusion. The latter model trained on more data took a longer time to achieve steady mAP over the iterations. Both model performed better than expected at the best validation of mAP ~98%.

## mAP vs IOU treshhold

As IOU treshhold increases, average precision drops. However, the latter model that was trained on a newer dataset performed significantly better at predicting `mask_weared_incorrect`. This is of course expected as the newer weights have learnt from more examples. The results shows that by introducing more data on inblanaced classes it will improve the overall prediction of the model.

![Old_mask_1](https://github.com/kmt112/probable-lamp/blob/master/results/table%20results.PNG)

## Prediction of sample images

Exhibit 1. Validation on old weights (left) and new weight (right)

![Old_mask_1](https://github.com/kmt112/probable-lamp/blob/master/results/Mask_incorrect_29_old.jpg) ![New mask_1](https://github.com/kmt112/probable-lamp/blob/master/results/Mask_incorrect_29_new.jpg)

Exhibit 2. Validation on old weights (left) and new weights (right)

![Old_mask_2](https://github.com/kmt112/probable-lamp/blob/master/results/Mask_incorrect_99_old.jpg) ![New mask_2](https://github.com/kmt112/probable-lamp/blob/master/results/mask_incorrect_99_new.jpg)

Exhibit 3. Validation on old weights (left) and new weights (right)

<img src="https://github.com/kmt112/probable-lamp/blob/master/results/mask%20incorect%20(old).png" width="400" height="500"> <img src="https://github.com/kmt112/probable-lamp/blob/master/results/mask%20incorect%20(New).jpg" width="400" height="500">

The various exhibits show that with the inclusion of new data, especially on the inbalanced classes, greatly improves the precision of the model.

## License
MIT
Darknet Yolov4
