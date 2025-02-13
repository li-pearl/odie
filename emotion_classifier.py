import cv2
import torch
import numpy as np
from fer import FER

# Initialize Face Detector & Emotion Recognizer
detector = FER()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotion using FER
    results = detector.detect_emotions(frame)

    for result in results:
        (x, y, w, h) = result["box"]
        emotion, score = max(result["emotions"].items(), key=lambda item: item[1])

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display emotion classification
        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Emotion Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
