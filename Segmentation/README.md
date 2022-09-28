# MMCS-Net
This repo holds code for [Combining Photogrammetric Computer Vision and Semantic Segmentation for Fine-grained Understanding of Coral Reef Growth Variations]

## Usage

### 1. Preparation

Please go to ["./datasets/README.md"](datasets/README.md) for details.

Please go to ["./utils/mypath.py] to modify the path.

### 2. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Train/Test
- Run the train script on coral dataset. The data is the excerpt of the whole dataset, 

```bash
python train.py --dataset plot18_2019 --backbone resnet_rgbd --use_balanced_weights --kfoldval True
```

## Model
MMCS-Net
![avatar](./img/img2.png)


## Result
![avatar](./img/img1.png)

## Reference
* [Deeplab V3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
