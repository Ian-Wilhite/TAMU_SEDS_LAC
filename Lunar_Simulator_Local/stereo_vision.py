#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:12:40 2025

@author: nitaishah
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

imgL = cv2.imread("/Users/nitaishah/Desktop/Lunar_Simulator_Imamages/l30.png")
imgR = cv2.imread("/Users/nitaishah/Desktop/Lunar_Simulator_Imamages/r30.png")

imgL = cv2.normalize(imgL, None, 0, 255, cv2.NORM_MINMAX)
imgR = cv2.normalize(imgR, None, 0, 255, cv2.NORM_MINMAX)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title(" Left Image")
plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Right Image")
plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

image_width = 1280
image_height = 720

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=16*16, blockSize=29)

disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# Display the disparity map
plt.figure(figsize=(10, 5))
plt.title("Disparity Map")
plt.imshow(disparity, cmap='gray')
plt.colorbar(label="Disparity Value")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(disparity[disparity > 0].ravel(), bins=50, color='blue', alpha=0.7)
plt.title("Disparity Value Distribution")
plt.xlabel("Disparity Value")
plt.ylabel("Frequency")
plt.show()

baseline = 0.162
focal_length = 915.7

disparity[disparity <= 0] = 0.1

disparity.max()


# Compute depth map
depth_map = (focal_length * baseline) / disparity

depth_map.min()
depth_map.max()
depth_map.shape
depth_map.std()


# Visualize the depth map
plt.figure(figsize=(10, 5))
plt.title("Depth Map")
plt.imshow(depth_map, cmap='jet')
plt.colorbar(label="Depth (meters)")
plt.axis("off")
plt.show()

def clean_inverse_depth_map(depth_map, limit=2000):
    """
    Cleans an inverse depth map by setting values representing depth greater than the limit to 0.

    Parameters:
        depth_map (np.ndarray): A 2D array representing the inverse depth map.
        limit (float): The threshold depth value. Depth values greater than this will be set to 0.

    Returns:
        np.ndarray: The cleaned inverse depth map.
    """
    # Ensure the input is a NumPy array
    depth_map = np.array(depth_map)
    
    # Assuming that lower numerical values represent greater depths
    # You might need to map pixel intensities to actual depth values first
    # For example, if pixel intensity 0 = depth 4000 and 255 = depth 0,
    # you need to convert pixel values to actual depth before applying the limit
    
    # Example conversion (this will vary based on your specific mapping)
    # Here, assuming a linear mapping for demonstration:
    max_pixel_value = 255
    max_depth = 6237  # Example maximum depth
    depth_values = (max_depth / max_pixel_value) * depth_map  # Convert pixel to depth
    
    # Set depth values greater than the limit to 0
    depth_values = np.where(depth_values > limit, 0, depth_values)
    
    # Optionally, convert back to pixel values if needed
    cleaned_pixel_map = (depth_values * (max_pixel_value / max_depth)).astype(depth_map.dtype)
    
    return cleaned_pixel_map

cleaned_map = clean_inverse_depth_map(depth_map, limit=500)
plt.imshow(cleaned_map)    

h, w = cleaned_map.shape

# Create pixel coordinate grid
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Compute 3D coordinates
X = (u - 640) * cleaned_map / focal_length
Y = (v - 360) * cleaned_map / focal_length
Z = cleaned_map

# Stack coordinates to form a point cloud
points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)


# Optional: Visualize the point cloud using Open3D
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([point_cloud])
o3d.io.write_point_cloud("point_cloud_1.ply", point_cloud)
