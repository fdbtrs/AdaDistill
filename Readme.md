## This is the official repository of the paper:
### AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition

### Accepted at ECCV 2024 (main conference)
[Arxiv](https://arxiv.org/abs/2407.01332)

![Poster](https://raw.githubusercontent.com/fdbtrs/AdaDistill/main/img/Poster.png)


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




If you use any of the code provided in this repository, please cite the following paper:
## Citation
```
@InProceedings{Boutros_2024_ECCV,
    author    = {Fadi Boutros, Vitomir Štruc, Naser Damer},
    title     = {AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
    booktitle = {Computer Vision - {ECCV} 2024 -18th European Conference on Computer Vision, Milano, Italy, September 29- 4 October, 2024 },
    month     = {October},
    year      = {2024},
    pages     = {}
}


```


## License

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2024 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```