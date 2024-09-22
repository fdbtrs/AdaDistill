## This is the official repository of the paper ECCV 2024:
#### AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition

### Accepted at ECCV 2024 (main conference)

![Poster](https://raw.githubusercontent.com/fdbtrs/AdaDistill/main/img/Poster_thumbnail.png)


## Installation
- The code is running using Pytorch 1.7
- Download the requirements from the file requirement.txt
- Download the processed MS1MV2 from the [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_), unzip it and place it inside the data folder
- The code is originally designed to run on 4 GPUs which can be changed from running scripts, run_standalone.sh 

## Teacher and standalone student training
- Set the config.network parameter in the config/config.py to iresent50 or mobilefacenet
- Set the output folder where the model should be saved
- The teacher and standalone student can be trained by running ./run_standalone.sh

## ArcDistill and CosDistill training
- Set the config.network parameter in the config/config.py to mobilefacenet
- Set the output folder where the model should be saved in the config/config.py 
- Set the path to the teacher header and backbone from previous training in the config/config.py , config.pretrained_teacher_path  and config.pretrained_teacher_header_path 
- Set the penalty loss to ArcFace or to CosFace in the config/config.py 
- ArcDistill and CosDistill can be trained by running ./run_AMLDistill.sh


## AdaDistill training
- Set the config.network parameter in the config/config.py to mobilefacenet
- Set the parameter config.adaptive_alpha in the config/config.py to **False**
- Set the output folder where the model should be saved in the config/config.py 
- Set the path to the teacher backbone in the config/config.py, config.pretrained_teacher_path
- Set the penalty loss to ArcFace or to CosFace in the config/config.py
- AdaArcDistill and AdaCosDistill can be trained by running ./run_AdaDistill.sh


## AdaDistill training with weighted alpha
- Set the config.network parameter in the config/config.py to mobilefacenet
- Set the parameter config.adaptive_alpha in the config/config.py to **True**
- Set the output folder where the model should be saved in the config/config.py 
- Set the path to the teacher backbone in the config/config.py, config.pretrained_teacher_path
- Set the penalty loss to ArcFace or to CosFace in the config/config.py
- AdaArcDistill and AdaCosDistill can be trained by running ./run_AdaDistill.sh


### <font color="green">A trained MobileFaceNet model with AdaArcDsitll is provided under output/AdaDistill/MFN_AdaArcDistill_backbone.pth</font>