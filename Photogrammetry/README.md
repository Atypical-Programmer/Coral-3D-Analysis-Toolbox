
Method
---------------

## Feature Matching
The SIFT is modified as RootSIFT.
The installation of AdaLAM follows [AdaLAM](https://github.com/cavalli1234/AdaLAM).

## Reconstruction
The 3D reconstruction is based on [OpenDroneMap](https://opendronemap.org/). It is a program that runs in the command line. Its release can be found [here](https://github.com/OpenDroneMap/ODM/releases).

Here is an example:
```bash
run PATH_TO_PROJECT --gcp PATH_TO_GCPFIEL --dsm
```

The orthophotos/DOMs and DSMs can be downloaded from [here](https://drive.google.com/drive/folders/18vh2BUHEDRaTmuHEKyhz3XT1_jmJSWPg?usp=sharing).
![avatar](./img/img1.png)

## Roughness
[Benthic Terrain Modeler (BTM) 3.0](https://www.arcgis.com/home/item.html?id=b0d0be66fd33440d97e8c83d220e7926#!?TB_iframe=true) is a toolbox for ArcGIS, and is used to calculate Vector Ruggedness Measure (VRM).