# Voxel2Hemodynamics
# Introduction
This is the repo of the paper Voxel2Hemodynamics: An End-to-end Deep Learning Method for Predicting Coronary Artery Hemodynamics.
# Content 
This repo contains an end to end framework to get the hemodynamics prediction of coronary artery given a dcm image. First we
generate pointcloud data from dcm and then use pointnet++ to predict the hemodynamics. The evaluation code is provided, and the training code
will be updated soon.
# Setup
```
cd Voxel2Hemodynamics
pip install -r requirement.txt
```
# Directory Structure
```
Voxel2Hemodynamic
├── checkpoints
    ├── Tag-VesselMeshSegmentation-GCN-best-checkpoint.pth(weight for segmentation and vectorization)
    ├── model_best.pth(weight for Hemodynamic Prediction)
├── data
    ├── mcrops
    ├── test_data
├── data_utils
├── README.md
├── models
    ├── pointnet2_sem_seg.py(pointnet++)
├── DDP_test.sh
├── config-predict.yaml
└── mtools(code from another paper, you don't need to see or modify it)
```
# Visualization 
After the prediction, the visualization result of the prediction can be get using the software Paraview
the link for download this software is: https://www.paraview.org/
