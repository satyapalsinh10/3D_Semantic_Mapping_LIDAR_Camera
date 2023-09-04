import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
import struct
import copy
from Code.Utils import *
import sys
sys.path.insert(0, '/home/mandeep/Lidar/Code/DeepLabV3Plus-Pytorch')
from predict import semantics
from argparse import ArgumentParser
import shutil

def main():

    #caluclate the transformation from LiDAR to camera
    p_lidar_to_cam = Projection_lidar_to_cam()

    save_path_img = os.path.join('Results', 'projected_clouds')
    make_directory(save_path_img)

    save_path_cloud = os.path.join('Results', 'painted_clouds')
    make_directory(save_path_cloud)

    rgb_folder_path = os.path.join('Code', 'Data', 'rgb_data')
    pcd_folder_path = os.path.join('Code', 'Data', 'pointcloud')
    img_list= os.listdir(rgb_folder_path)
    img_list.sort()
    pcd_list = os.listdir(pcd_folder_path)
    pcd_list.sort()

    img_paths = []
    pcd_paths = []

    for a in range(len(img_list)):

        img_paths.append(os.path.join(rgb_folder_path, img_list[a]))
        pcd_paths.append(os.path.join(pcd_folder_path, pcd_list[a]))

    count = 10000
    for i in range(len(img_list)):

        img_path = img_paths[i]
        img = cv2.imread(img_path)
        pcd = o3d.io.read_point_cloud(pcd_paths[i])
        pcd_array = np.asarray(pcd.points)

        # remove all the points from the point cloud that are behind the camera (means backside of the car which is -ve x axis)
        idx = pcd_array[:,0] >= 0 
        pcd_array= pcd_array[idx]
        #projet lidar points on RGB image
        pts_2D,depth, pts_3D_img = project_lidar_on_image(p_lidar_to_cam, pcd_array, (img.shape[1], img.shape[0]))
        N = pts_3D_img.shape[0]

        #predict semantic segmentation of RGB images using DeepLavV3+ pretrained model
        pred, semantic_rgb = semantics(img_path)
        cloud_color = np.zeros((N,3), dtype=np.float32)
        fused_img = img.copy()
        for j in range(pts_2D.shape[0]):
            if j >= 0:

                x = np.int32(pts_2D[j, 0])
                y = np.int32(pts_2D[j, 1])
                
                # get the color corresponsing to the label
                class_color = np.float64(semantic_rgb[y, x]) 
                #draw the point cloud projections on the image
                cv2.circle(fused_img, (x,y), 2, color=tuple(class_color), thickness=1)

                # assign color to point clouds
                cloud_color[j] = class_color/255.0

        stacked_img = np.vstack((img,fused_img))
        cv2.imwrite(save_path_img + "/" + str(count) + ".png",stacked_img)

        # colored point cloud
        semantic_pointcloud = np.hstack((pts_3D_img[:,:3], cloud_color))

        # visualize point clouds
        visualize_pointcloud(semantic_pointcloud, count, save_path_cloud)
        
        count+=1

    video_file = "video.mp4"
    make_video(10, save_path_img, video_file)

if __name__ == '__main__':
    main()