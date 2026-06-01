import cv2
from ultralytics import YOLO

# Load pose model
model = YOLO("yolov8n-pose.pt")

# Open video
cap = cv2.VideoCapture("walking people.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Pose estimation
    results = model(frame)

    # Draw skeletons automatically
    annotated_frame = results[0].plot()
    display_frame = cv2.resize(annotated_frame, (1000, 600))
    cv2.imshow("Pose Estimation", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()