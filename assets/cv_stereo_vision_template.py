import cv2
import numpy as np
import ORB_SLAM2
import open3d as o3d

# Load ORB-SLAM2 System (path to vocabulary and configuration file)
slam = ORB_SLAM2.System("ORBvoc.txt", "config/stereo.yaml", ORB_SLAM2.Sensor.STEREO)
slam.initialize()

# Stereo Camera Parameters (Adjust according to your setup)
fx = 718.856  # Focal length x
fy = 718.856  # Focal length y
cx = 607.1928  # Principal point x
cy = 185.2157  # Principal point y
baseline = 0.54  # Baseline distance between stereo cameras (meters)

# List of stereo image pairs (Modify this to get images from a real source)
stereo_pairs = [("left1.jpg", "right1.jpg"), ("left2.jpg", "right2.jpg")]  

for left_img_path, right_img_path in stereo_pairs:
    # Read stereo images
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        print(f"Error loading images: {left_img_path}, {right_img_path}")
        continue

    # Pass images to ORB-SLAM2
    Tcw = slam.process_stereo(left_img, right_img, timestamp=0.0)  # Timestamp can be real time

# Shutdown SLAM and save results
slam.shutdown()

# Save SLAM Map
slam.save_map("output/orb_slam_map.bin")
slam.save_trajectory("output/orb_trajectory.txt")
print("ORB-SLAM2 map and trajectory saved.")




import numpy as np
import open3d as o3d

# Load trajectory data (KITTI format)
trajectory_data = np.loadtxt("output/orb_trajectory.txt")

# Extract XYZ positions
positions = trajectory_data[:, 1:4]

# Create Open3D Point Cloud
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(positions)

# Save as PLY
o3d.io.write_point_cloud("output/orb_slam_map.ply", pc)
print("ORB-SLAM2 point cloud saved as PLY.")



import open3d as o3d

# Load the SLAM-generated point cloud
pc = o3d.io.read_point_cloud("output/orb_slam_map.ply")

# Visualize
o3d.visualization.draw_geometries([pc])
