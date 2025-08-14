import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import time
import os

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable higher resolution and FPS if supported
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

# Start streaming
try:
    pipeline.start(config)
except RuntimeError as e:
    print(f"Pipeline start failed: {e}")
    print("Try lowering resolution or checking camera connection.")
    exit()

# Create an align object to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Get depth scale
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # Convert depth values to meters

# Apply RealSense filters for better accuracy
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

# Define the output file path in the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
output_file = os.path.join(project_dir, "point_cloud.txt")  # Save in the project folder

# Open file for writing (overwrite if exists) and write header once
with open(output_file, "w") as f:
    f.write("X Y Z\n")  # Write header

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("RealSense Point Cloud", width=1280, height=720)

# Create a point cloud object
pcd = o3d.geometry.PointCloud()
added = False

header_printed = False  # Ensure "X Y Z" header is only printed once

try:
    while True:
        start_time = time.time()  # Measure loop time for performance tuning

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Apply depth post-processing filters
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color image from BGR to RGB
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Create intrinsic object
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Create Open3D camera intrinsic object
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        )

        # Convert depth image to float32 (in meters)
        depth_image_meters = depth_image.astype(np.float32) * depth_scale

        # Convert images to Open3D format
        depth_image_o3d = o3d.geometry.Image(depth_image_meters)
        color_image_o3d = o3d.geometry.Image(color_image_rgb)

        # Create RGBD image with extended depth range
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image_o3d, depth_image_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        # Generate point cloud from RGBD image
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsic
        )

        # Flip the point cloud to correct the orientation
        temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Apply voxel down sampling to reduce the number of points
        voxel_size = 0.01  # Adjust this value to control the density of points
        temp_pcd = temp_pcd.voxel_down_sample(voxel_size=voxel_size)

        # Extract down sampled XYZ coordinates
        xyz_points = np.asarray(temp_pcd.points)

        # Ensure we do not exceed 10,000 points
        if xyz_points.shape[0] > 10000:
            xyz_points = xyz_points[np.random.choice(xyz_points.shape[0], 10000, replace=False)]

        # Convert to a matrix-like string format
        matrix_str = np.array2string(xyz_points, separator=', ', threshold=np.inf)

        # Print header only once
        if not header_printed:
            print("X Y Z")
            header_printed = True  # Ensure header is not printed again

        # Print the formatted output (no repeated "X Y Z")
        print(matrix_str)

        # Save the matrix format to the file
        with open(output_file, "a") as f:
            f.write(f"{matrix_str}\n")

        # Update the point cloud
        pcd.points = temp_pcd.points
        pcd.colors = temp_pcd.colors

        if not added:
            vis.add_geometry(pcd)
            added = True

        # Update the visualizer efficiently
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Print FPS for debugging
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

finally:
    # Stop streaming
    pipeline.stop()
    vis.destroy_window()
    print(f"Point cloud saved to: {output_file}")
