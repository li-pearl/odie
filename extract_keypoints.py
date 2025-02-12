import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

# Load MoveNet model
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
model = movenet.signatures['serving_default']

# CSV file to store pose data
CSV_FILE = "pose_data.csv"
POSE_LABELS = ["Standing", "Sitting", "Leaning", "Hand Raised"]  # Modify as needed
pose_data = []

# Function to preprocess image for MoveNet
def preprocess_image(frame):
    img = cv2.resize(frame, (192, 192))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_img = preprocess_image(frame)
    outputs = model(tf.constant(input_img))  # Run MoveNet inference
    keypoints = outputs["output_0"].numpy().reshape(17, 3)  # Extract keypoints

    # Ask user for label
    print("Available Labels:", POSE_LABELS)
    pose_label = input("Enter the pose label: ")  # User labels the pose

    if pose_label in POSE_LABELS:
        pose_data.append(keypoints.flatten().tolist() + [pose_label])

    # Draw keypoints
    for x, y, conf in keypoints:
        if conf > 0.5:
            x, y = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("MoveNet Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save keypoints to CSV
df = pd.DataFrame(pose_data)
df.to_csv(CSV_FILE, index=False)
print(f"Pose keypoints saved to {CSV_FILE}")
