# Voxel2Hemodynamics
# Introduction
This is the repo of the paper Voxel2Hemodynamics: An End-to-end Deep Learning Method for Predicting Coronary Artery Hemodynamics.
# Content 
This repo contains an end to end framework to get the hemodynamics prediction of coronary artery given a dcm image. First we
generate pointcloud data from dcm and then use pointnet++ to predict the hemodynamics. The evaluation code is provided, and the training code
will be updated soon.
# Directory Structure
```
Voxel2Hemodynamic
├── checkpoints
├── data
├── data_utils
├── README.md
├── models
├── DDP_tesh.sh
├── config-predict.yaml
└── mtools
```
