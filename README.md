# pcd_ground_segmentation
<img src=misc/sample_img.png></img>
- This repository contains the trained model and code(train/evaluation/inference) to <b> segment the area corresponding to the ground on the point cloud.</b>
- Reimplementation of original paper <i>"CNN for Very Fast Ground Segmentation in Velodyne LiDAR Data (2017)"</i>

<br>
<br>

# Semantic KITTI Baseline
``` 
Acc avg 0.867
IoU avg 0.594
Recall avg 0.867
IoU class 0 [unlabeled] = 0.404
IoU class 1 [car] = 0.367
IoU class 2 [bicycle] = 0.213
IoU class 3 [motorcycle] = 0.446
IoU class 4 [truck] = 0.898
IoU class 5 [other-vehicle] = 0.725
IoU class 6 [person] = 0.380
IoU class 7 [bicyclist] = 0.398
IoU class 8 [motorcyclist] = 0.272
>> IoU class 9 [road] = 0.986
IoU class 10 [parking] = 0.075
>> IoU class 11 [sidewalk] = 0.960
>> IoU class 12 [other-ground] = 0.986
IoU class 13 [building] = 0.920
IoU class 14 [fence] = 0.714
IoU class 15 [vegetation] = 0.768
IoU class 16 [trunk] = 0.706
>> IoU class 17 [terrain] = 0.897
IoU class 18 [pole] = 0.367
IoU class 19 [traffic-sign] = 0.400

(>> : ground class)
```
* How to assign point labels
```
ground_label = [9, 11, 12, 17]
is_ground = np.zeros_like(pred)
is_ground[np.isin(label, ground_label)] = 1

# accuracy with original label
pred = np.where(np.logical_and(pred == 1, is_ground == 1), label, pred)
pred = np.where(np.logical_and(pred == 0, is_ground == 0), label, pred)
```

<br>

## visualize
<img src=misc/predict_1.png></img>
<img src=misc/predict_2.png></img>


<br>
<br>

# Getting Started
## Dataset
- In This project, Aimmo dataset format is not supproted. Only Semantic KITTI dataset can import.
- Download the [Semantic KITTI dataset](http://www.semantic-kitti.org/dataset.html) and organize the downloaded files as official directory tree.
```
dataset
└── sequences
    ├── 00
    │   ├── labels
    │   └── velodyne
    ├── 01
    │   ├── labels
    │   └── velodyne
    ...
```
- Setting config.yaml file for dataset path
```
path:
  data_root: "/home/ext/SemanticKITTI/dataset"
  label_root: "/home/ext/SemanticKITTI/dataset"
  pcd_root: "/home/ext/SemanticKITTI/dataset"
  
```
<br>

## Model files
For training model, pretrained model is not supported in this project.   
You can [download](https://drive.google.com/file/d/1ArfgjplSinrlJRE8b29vmAYkqqU6PCDC/view?usp=share_link) trained model.

<br>

## Training & Evaluation
train   
`python train.py --cfg config.yaml`


evaluation     
`python eval.py --cfg config.yaml --ckpt experiments/14_srate15-interpolate/ckpts/10.pth`   

<br>

## Inference
```
import inference

infr = inference.Inference(cfg_path='config.yaml', device='gpu')
infr.load_model('experiments/14_srate15-interpolate/ckpts/10.pth')
path = ['/home/ext/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin', ...]
result = infr.predict(path)

print(len(result[0]))
>>115295
```


---
@ AI-Lab milla
