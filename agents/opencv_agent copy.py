#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import carla
import cv2 as cv
import random
from math import radians
from pynput import keyboard
import open3d as o3d

""" Import the AutonomousAgent from the Leaderboard. """
from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """
def get_entry_point():
    return 'OpenCVagent'

""" Inherit the AutonomousAgent class. """
class OpenCVagent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        """ This method is executed once by the Leaderboard at mission initialization. """
        
        """ Set up a keyboard listener from pynput to capture the key commands. """
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """
        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.frame = 0

    def use_fiducials(self):
        """ We want to use the fiducials, so we return True. """
        return True

    def sensors(self):
        """ Define sensor configurations. """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
        }
        return sensors

    def run_step(self, input_data):
        """ Process the sensor input and generate depth map and point cloud. """
        
        if self.frame == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
        
        # Debugging: Check available input data
        print("Received input data keys:", input_data.keys())

        # Get left and right camera images safely
        left_image = input_data.get('Grayscale')
        right_image = input_data.get('Grayscale')


        # Ensure that images exist before proceeding
        if left_image is None or right_image is None:
            print("Error: One or both camera images are missing!")
            return carla.VehicleVelocityControl(0, 0)

        try:
            image_width, image_height = left_image.shape[0], right_image.shape[0]

            # Convert CARLA image format to OpenCV format
            left_cv = np.array(left_image.raw_data).reshape((image_height, image_width, 4))[:, :, :3]
            right_cv = np.array(right_image.raw_data).reshape((image_height, image_width, 4))[:, :, :3]

            # Convert images to grayscale
            grayL = cv.cvtColor(left_cv, cv.COLOR_BGR2GRAY)
            grayR = cv.cvtColor(right_cv, cv.COLOR_BGR2GRAY)

            # Compute Disparity Map
            stereo = cv.StereoBM_create(numDisparities=16 * 16, blockSize=29)
            disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

            # Compute Depth Map
            baseline = 0.162  # meters
            focal_length = 915.7  # pixels
            disparity[disparity <= 0] = 0.1  # Avoid division by zero
            depth_map = (focal_length * baseline) / disparity

            # Convert Depth Map to 3D Point Cloud
            point_cloud = self.generate_point_cloud(depth_map, left_cv)

            # Save or visualize point cloud
            o3d.io.write_point_cloud("output.ply", point_cloud)
            o3d.visualization.draw_geometries([point_cloud])

        except Exception as e:
            print(f"Error in processing: {e}")
            return carla.VehicleVelocityControl(0, 0)

        return carla.VehicleVelocityControl(0, 0)
    
    def generate_point_cloud(self, depth_map, color_image):
        """ Generate a 3D point cloud from depth map. """
        h, w = depth_map.shape
        fx, fy = 915.7, 915.7  # Focal lengths
        cx, cy = w // 2, h // 2  # Optical center
    
        points = []
        colors = []
    
        for y in range(h):
            for x in range(w):
                depth = depth_map[y, x]
                if depth > 0 and depth < 100:  # Ignore invalid depths
                    X = (x - cx) * depth / fx
                    Y = (y - cy) * depth / fy
                    Z = depth
    
                    points.append([X, Y, Z])
                    colors.append(color_image[y, x] / 255.0)  # Normalize colors
    
        # Convert to Open3D point cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.array(points))
        pc.colors = o3d.utility.Vector3dVector(np.array(colors))
    
        return pc

    def finalize(self):
        """ Clean up OpenCV resources. """
        cv.destroyAllWindows()

    def on_press(self, key):
        """ Keyboard controls for manual override. """
        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6      

    def on_release(self, key):
        """ Reset controls when keys are released. """
        if key in [keyboard.Key.up, keyboard.Key.down]:
            self.current_v = 0
        if key in [keyboard.Key.left, keyboard.Key.right]:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()
