# MMCS-Net
This repo holds code for [Combining Photogrammetric Computer Vision and Semantic Segmentation for Fine-grained Understanding of Coral Reef Growth Variations]

## Usage

### 1. Preparation

Please go to ["./datasets/plot18_2019/README.md"](./dataset/plot18_2019/README.md) for details.

Please go to ["./utils/mypath.py] to modify the path.

### 2. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Train/Test
- Run the train script on coral dataset. The data is the excerpt of the whole dataset, 

```bash
python train.py --dataset plot18_2019 --backbone resnet_rgbd --use_balanced_weights --kfoldval True
```

A sampled dataset can be downloaded from [here](https://drive.google.com/file/d/1T6cV43Uo5k5UbqJegOfwvHL-3MM0VEE8/view?usp=sharing).

## Model
MMCS-Net
![avatar](./img/img2.png)

## Description
To generate the dataset for training, the orthophoto is clipped into a set of patches, each of which is extracted from a sliding window swiped over the whole tile at a stride of 224 pixels. This approach guarantees that all pixels at the edge of a patch become central pixels in subsequent patches，as Diakogiannis did Remote Sensing Image Segmentation  in 2020[1]. 

[metrics.py](./utils/metrics.py) is for evaluation. In the validation procedure, a five-fold cross-validation is performed. The data used to train the model are split into five equal parts (folds). Each fold was used once as a validation while the remaining folds were used to train and run the model. The model was run five times and each time the accuracy and loss were calculated. As for the evaluation metrics, the results are reported using Mean Pixel Accuracy (mPA) and Mean Region Intersection over Union (mIoU).


## Result
![avatar](./img/img1.png)
* Model A - DeepLabv3+
* Model B - DeepLabv3+ (RGB-D)
* Model C - DeepLabv3+ (ShapeConv)

| Models   |      mPA      |  mIoU |
|----------|:-------------:|------:|
| Model A |  89.9% | 80.5% |
| Model B |    90.8%   |  82.1% |
| Model C | 91.6% |  83.5% |
| MMCS-Net | 92.2% |  84.7% |

## Reference
1. Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020). ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, 94-114.
1. [Deeplab V3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
