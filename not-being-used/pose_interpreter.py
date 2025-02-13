import torch
import torchvision
import cv2
import numpy as np

# Keypoint R-CNN model (COCO dataset)
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval() 

# preprocess frame for model input
def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
    img = torch.tensor(img).unsqueeze(0)  # Add batch dimension
    return img

def draw_keypoints(frame, keypoints):
    height, width, _ = frame.shape

    for person_kp in keypoints:
        for kp in person_kp:  # Each person has 17 keypoints
            x, y, score = kp
            if score > 0.5:  # Confidence threshold
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame for model input
    img_tensor = process_frame(frame)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    keypoints = outputs["keypoints"].numpy()  # Shape: (N, 17, 3)

    # Draw keypoints on frame
    draw_keypoints(frame, keypoints)

    # Display
    cv2.imshow("PyTorch Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
