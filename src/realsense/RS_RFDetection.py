# Realsense connected to Roboflow model to detect a person
import cv2
import threading
import numpy as np
import pyrealsense2 as rs
from roboflow import Roboflow

# Initialize Roboflow and load model
rf = Roboflow(api_key="YJCg5JNCeL95vMzwd6kY")
project = rf.workspace().project("human-finder-pzjbe-eedl1")
model = project.version(1).model

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)  # Adjust resolution & FPS as needed

# Start pipeline
pipeline.start(config)

frame_skip = 2  # Process every nth frame
frame_count = 0
latest_frame = None
predictions = {"predictions": []}
lock = threading.Lock()


def run_inference():
    """Run object detection inference in a separate thread."""
    global predictions, latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()
        try:
            preds = model.predict(frame_copy, confidence=40, overlap=30).json()
            with lock:
                predictions = preds if isinstance(preds, dict) else {"predictions": []}
        except Exception as e:
            print(f"API Error: {e}")
            with lock:
                predictions = {"predictions": []}

# Start inference thread
thread = threading.Thread(target=run_inference, daemon=True)
thread.start()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())  # Convert to numpy array

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Update latest frame for inference
        with lock:
            latest_frame = frame.copy()

        # Draw bounding boxes from last predictions
        with lock:
            for prediction in predictions.get('predictions', []):
                x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(
                    prediction['height'])
                label = prediction['class']
                confidence = prediction['confidence']

                # Draw bounding box and label
                cv2.rectangle(frame, (x - width // 2, y - height // 2),
                              (x + width // 2, y + height // 2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                            (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Human Detection - RealSense", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and close windows
    pipeline.stop()
    cv2.destroyAllWindows()


