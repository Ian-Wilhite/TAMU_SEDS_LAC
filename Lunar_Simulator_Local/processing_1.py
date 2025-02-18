#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:43:06 2025

@author: nitaishah
"""

import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay

# Load the point cloud (replace with your file)
pcd = o3d.io.read_point_cloud("/Users/nitaishah/Desktop/Point Cloud Processing/Generated_Point_Clouds/cleaned_point_cloud_features_9.ply")  # .ply, .pcd, or .xyz
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

# Orient normals consistently towards the camera
pcd.orient_normals_consistent_tangent_plane(100)

# Visualize normals (set normal length for better visibility)
o3d.visualization.draw_geometries([pcd], window_name="Point Normals", point_show_normal=False)


mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Compute vertex normals
mesh.compute_vertex_normals()

# Visualize the generated mesh
o3d.visualization.draw_geometries([mesh], window_name="Poisson Surface Mesh")


vertices = np.asarray(mesh.vertices)

# Get Z-values (elevation)
z_values = vertices[:, 2]

# Normalize Z-values to range [0, 1] for color mapping
z_min, z_max = np.min(z_values), np.max(z_values)
z_norm = (z_values - z_min) / (z_max - z_min)

# Apply color gradient (Red = High, Blue = Low)
colors = np.zeros((len(vertices), 3))  # RGB array
colors[:, 0] = z_norm  # Red channel (higher elevation → more red)
colors[:, 2] = 1 - z_norm  # Blue channel (lower elevation → more blue)

# Assign vertex colors
mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# Visualize the elevation-colored mesh
o3d.visualization.draw_geometries([mesh], window_name="Elevation-Colored Terrain")

o3d.io.write_triangle_mesh("elevation_colorized_lunar_terrain.ply", mesh)

pcd = o3d.io.read_point_cloud("elevation_colorized_lunar_terrain.ply")

plane_model, inliers = pcd.segment_plane(distance_threshold=0.008,
                                         ransac_n=3,
                                         num_iterations=1000)

# Extract ground (inliers) and non-ground (outliers)
ground_pcd = pcd.select_by_index(inliers)
non_ground_pcd = pcd.select_by_index(inliers, invert=True)

# Colorize for better visualization
ground_pcd.paint_uniform_color([0.6, 0.3, 0.1])  # Brownish ground
non_ground_pcd.paint_uniform_color([0, 0, 1])  # Blue non-ground

# Visualize the separated ground vs non-ground
o3d.visualization.draw_geometries([ground_pcd, non_ground_pcd], window_name="Ground vs Non-Ground")

o3d.io.write_point_cloud("lunar_ground_ransac.ply", ground_pcd)
o3d.io.write_point_cloud("lunar_non_ground_ransac.ply", non_ground_pcd)


o3d.visualization.draw_geometries([non_ground_pcd], window_name="Elevation-Colored Terrain")
