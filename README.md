# Semantic Mapping with LIDAR and Camera Data

This repository contains the code implementing Semantic Mapping using LIDAR and Camera data. The project involves building a high-definition map from raw LIDAR point cloud data and transferring semantic labels from RGB images onto the LIDAR point cloud. The goal is to create a map representing the environment and label objects with their semantic information. Each 3D point in the point cloud has been assigned a label using the RGB semantic segmentation performed on the KITTI-360 dataset by the DeepLabV3+ network using extrinsic from the camera and LiDAR.


## Usage Guidelines:

### 1. Get the pre-trained model

To do semantic segmentation of RGB images, get the pre-trained model of DeepLabV3plus from [here](https://github.com/VainF/DeepLabV3Plus-Pytorch).

### 2. Download KiTTI-360 Dataset

Download the stereo camera images and Velodyne sensor data from any of the 9 sequences of the KITTI-360 dataset and store it in the `Code/Data` folder as specified. The dataset can be downloaded from [here](https://www.cvlibs.net/datasets/kitti-360/index.php).

### 3. Execution

 To execute enter the below command when in the parent folder of this repo.:
```
python3 Wrapper.py --path **path_to_dataset_folder**
```

## Results:

#### 1. Segmented RGB image and point cloud projections on the image.

In this part of the project, we achieved the following results:

- **Segmented RGB Image**: Utilizing a semantic segmentation neural network, we generated segmented RGB images where different objects and regions are labeled with their semantic information. These segmented images were a crucial part of our project for associating semantic labels with the corresponding 3D points in the LIDAR point cloud.

- **Point Cloud Projections on the Image**: To provide a visual understanding of how our semantic labels align with the 3D environment, we projected the LIDAR point cloud onto the segmented RGB images. This projection allowed us to visualize how objects in the point cloud correspond to the labeled regions in the RGB images.




<p align="center">
  <img src="Code/Results/lidar_semantics.gif"  align="center" alt="Undistorted" width="600"/>
</p>

#### 2. Semantically Segmented Point Cloud

In the second part of our project, we obtained semantically segmented point clouds. Here's what we achieved:

- **Semantically Labeled Point Cloud**: Leveraging the semantic labels generated from the RGB images, we successfully transferred these labels onto the LIDAR point cloud. This process resulted in a semantically labeled 3D point cloud where each point is associated with a semantic category.

- **Visual Verification**: To validate the correctness of our semantically segmented point cloud, we visually inspected the results. Objects and regions in the point cloud were colored according to their semantic labels. While there may be some minor errors due to imperfect semantic predictions, the overall mapping of semantics onto the point cloud was successful.



<p align="center">
  <img src="Code/Results/pointcloud.gif"  align="center" alt="Undistorted" width="600"/>
</p>

## Visualization
The Semantic_Mapping.mp4 video showcases the entire semantic mapping process. It includes visualizations of RGB images, 3D point clouds, and the transfer of semantic labels onto the LIDAR map.

## References

1. [PointPainting: Sequential Fusion for 3D Object Detection](https://arxiv.org/abs/1911.10150)
2. https://github.com/AmrElsersy/PointPainting
3. https://github.com/VainF/DeepLabV3Plus-Pytorch
4. https://rbe549.github.io/fall2022/hw/hw2
