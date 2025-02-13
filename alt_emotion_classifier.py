import cv2
import numpy as np

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Predefined colors for different emotions
emotion_model = {
    "happy": [(0, 255, 0)],  # Green
    "sad": [(255, 0, 0)],    # Blue
    "angry": [(0, 0, 255)],  # Red
    "neutral": [(255, 255, 255)],  # White
}

# Function to classify emotions (simplified)
def classify_emotion(face_roi):
    mean_intensity = np.mean(face_roi)  # Use pixel brightness as a simple feature
    if mean_intensity > 150:
        return "happy"
    elif mean_intensity < 80:
        return "sad"
    else:
        return "neutral"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Extract face region
        emotion = classify_emotion(face_roi)  # Run simple emotion classification

        # Fix: Ensure color is a tuple of three integers
        color = tuple(emotion_model.get(emotion, [(255, 255, 255)])[0])

        # Draw face box with color based on emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display detected emotion
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

    cv2.imshow("Emotion Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
