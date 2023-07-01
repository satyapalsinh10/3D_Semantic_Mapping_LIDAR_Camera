import numpy as np
import open3d
import cv2
import os
import matplotlib.pyplot as plt
import struct
import copy
from moviepy.editor import ImageSequenceClip
import shutil

def read_calib_data():
    filepath = "Code/calibration/calib.txt"
    file = open(filepath,"r")
    for i, row in enumerate(file):
        if i == 0:
            key, value = row.split(':')
            P = np.array([float(x) for x in value.split()]).reshape((3,4))
        if i == 1:
            key, value = row.split(':')
            K = np.array([float(x) for x in value.split()]).reshape((3,3))
        if i == 2:
            key, value = row.split(':')
            R0 = np.array([float(x) for x in value.split()]).reshape((3,3))
        if i == 3:
            key, value = row.split(':')
            Tr_cam_to_lidar = np.array([float(x) for x in value.split()]).reshape((3,4))
        if i == 4:
            key, value = row.split(':')
            D = np.array([float(x) for x in value.split()]).reshape((1,5))
    return P,K,R0,Tr_cam_to_lidar,D

def Projection_lidar_to_cam():
    P,K,R,Tr_cam_to_lidar,D = read_calib_data()
    R_cam_to_lidar = Tr_cam_to_lidar[:3,:3].reshape(3,3)
    t_cam_to_lidar = Tr_cam_to_lidar[:3,3].reshape(3,1)
    R_cam_to_lidar_inv = np.linalg.inv(R_cam_to_lidar)
    t_new = -np.dot(R_cam_to_lidar_inv , t_cam_to_lidar)
    Tr_lidar_to_cam = np.vstack((np.hstack((R_cam_to_lidar_inv, t_new)), np.array([0., 0., 0., 1.])))
    R_rect = np.eye(4)
    R_rect[:3, :3] = R.reshape(3, 3)
    P_ = P.reshape((3, 4))
    proj_mat = P_ @ R_rect  @ Tr_lidar_to_cam
    return proj_mat

def bin_to_pcd(bin_path, pcd_path):
    size_float = 4
    list_pcd = []
    with open(bin_path, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    open3d.io.write_point_cloud(pcd_path, pcd)
    # open3d.visualization.draw_geometries([pcd])
    # return pcd

# def bin_to_pcd_all():
#     folder_path = os.path.join('KITTI-360', 'data_3d_raw', '2013_05_28_drive_0000_sync', 'velodyne_points', 'data')
#     bin_list= os.listdir(folder_path)
#     save_path = "pcd_data"
#     for i in range(len(bin_list)):
#         print("count", i)
#         bin_path = os.path.join(folder_path,bin_list[i])
#         bin_to_pcd(bin_path, save_path + "/" + str(i) + ".pcd")

def visualize_pointcloud(pointcloud, count, save_path):

    xyz = pointcloud[:, 0:3]
    semantics = pointcloud[:,3:]

    #Initialize Open3D visualizer
    visualizer = open3d.visualization.Visualizer()
    pcd = open3d.geometry.PointCloud()
    visualizer.add_geometry(pcd)

    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(semantics)
    # open3d.visualization.draw_geometries([pcd])
    open3d.io.write_point_cloud(save_path + "/" + str(count) + ".pcd",pcd)

def project_lidar_on_image(P, lidar_pts, size):

    n = lidar_pts.shape[0]
    pts_3d =  np.hstack((lidar_pts, np.ones((n, 1))))        #homogeneous co-ordinates
    pts_2d = np.dot(pts_3d, P.T)
    depth = pts_3d[:,2]
    depth[depth==0] = -1e-6
    #normalize 3d points
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_2d = pts_2d[:, :2]
    #points inside the image frame
    inliers_idx = ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < size[0]) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < size[1]))
    return pts_2d[inliers_idx], depth[inliers_idx], lidar_pts[inliers_idx]

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def make_video(fps, path, video_file):
    """
    To make video from the images stored in an entire folder
    
    Inputs:
    fps : frame per second for the video
    path: folder path in which sequence of images are stored
    video_file: name of the video top be saved

    """
    print("Making video***********************")
    print("Creating video {}, FPS={}".format(video_file, fps))
    clip = ImageSequenceClip(path, fps = fps)
    clip.write_videofile(video_file)
    shutil.rmtree(path)


def make_directory(save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)