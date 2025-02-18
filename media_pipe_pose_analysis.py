import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

pose_label = ""

cap = cv2.VideoCapture(0)
pTime = 0

# Function to classify poses
def classify_pose(landmarks, img_shape):
    h, w, _ = img_shape  # Get image dimensions

    # Get keypoint coordinates
    nose = landmarks[0]  # Nose
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]  # Shoulders
    left_wrist, right_wrist = landmarks[15], landmarks[16]  # Wrists
    left_elbow, right_elbow = landmarks[13], landmarks[14]  # Elbows
    left_hip, right_hip = landmarks[23], landmarks[24]  # Hips

    # Convert keypoint positions to pixels
    ls_x, ls_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
    rs_x, rs_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
    lw_x, lw_y = int(left_wrist.x * w), int(left_wrist.y * h)
    rw_x, rw_y = int(right_wrist.x * w), int(right_wrist.y * h)
    le_x, le_y = int(left_elbow.x * w), int(left_elbow.y * h)
    re_x, re_y = int(right_elbow.x * w), int(right_elbow.y * h)
    lh_x, lh_y = int(left_hip.x * w), int(left_hip.y * h)
    rh_x, rh_y = int(right_hip.x * w), int(right_hip.y * h)
    
    # Calculate distances
    shoulder_width = abs(rs_x - ls_x)
    wrist_distance = abs(lw_x - rw_x)

    # Check if body is leaning
    lean_threshold = 20 
    body_tilt = abs(ls_y - rs_y)
    is_leaning = body_tilt > lean_threshold

    # Pose classification logic
    if lw_y < ls_y and rw_y < rs_y:
        return "Both Hands Raised"
    elif lw_y < ls_y:
        return "Left Hand Raised"
    elif rw_y < rs_y:
        return "Right Hand Raised"
    elif lw_x > rw_x and lw_y > lh_y:
        return "Arms Crossed"
    elif lw_x > rs_x and rw_x < ls_x:
        return "Arms Behind Back"
    elif wrist_distance > shoulder_width * 1.3:
        return "Open arms"
    elif is_leaning:
        return "Leaning"
    else:
        return "Undetected Pose"

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        pose_label = classify_pose(landmarks, img.shape)

        # for text display
        cv2.putText(img, pose_label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # fps calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Pose Classification", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()