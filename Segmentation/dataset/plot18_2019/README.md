# Data Preparing

1. Access to the coral dataset:
    You can send an Email directly to hqzhang@whu.edu.cn to request the whole dataset.
2. The directory structure of the dataset is as follows:

```bash
.
└── dataset
    └──polt18_2019
        └── data
        │    ├── images
        │    ├── masks (masks: (H,W,1) each value represents the class)
        │    ├── depths
        │    ├── train_k.txt (k means k-fold)
        │    ├── val_k.txt (k means k-fold)
        │    └── test_k.txt (k means k-fold)
        └── plot18_2019_pocill_classes_weights.npy
```
