import cv2
import numpy as np
from feat import Detector

# Initialize Py-Feat Detector
detector = Detector(
    face_model="retinaface", 
    landmark_model="mobilenet", 
    au_model="xgb", 
    emotion_model="resmasknet"
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        face_results = detector.detect_faces(frame)

        if face_results and len(face_results) > 0:
            for face in face_results:
                bbox = face["bbox"]  
                landmarks = face["landmarks"] 

                # Ensure bbox is correctly formatted as (x, y, w, h)
                x, y, w, h = map(int, bbox)

                emotion_scores = detector.detect_emotions(frame, facebox=bbox, landmarks=landmarks)
                
                if not emotion_scores.empty:
                    dominant_emotion = emotion_scores.idxmax(axis=1).values[0]  # Get most probable emotion
                else:
                    dominant_emotion = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Error: {e}")

    cv2.imshow("Py-Feat Emotion Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
