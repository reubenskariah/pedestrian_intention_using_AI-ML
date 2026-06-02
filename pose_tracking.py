from ultralytics import YOLO
import cv2
import csv

# Load YOLO Pose model
model = YOLO("yolov8n-pose.pt")

# Open video
cap = cv2.VideoCapture("walking people.mp4")

# Create CSV file
csv_file = open("trajectory.csv", "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "frame",
    "id",
    "nose_x",
    "nose_y"
])

frame_no = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_no += 1

    # Pose + Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],  # person only
        verbose=False
    )

    annotated_frame = results[0].plot()

    if (
        results[0].boxes.id is not None
        and results[0].keypoints is not None
    ):

        ids = results[0].boxes.id.cpu().numpy().astype(int)

        keypoints = results[0].keypoints.xy.cpu().numpy()

        for track_id, pose in zip(ids, keypoints):

            # Nose = keypoint 0
            nose_x = int(pose[0][0])
            nose_y = int(pose[0][1])

            # Save to CSV
            writer.writerow([
                frame_no,
                track_id,
                nose_x,
                nose_y
            ])

            print(
                f"Frame {frame_no} | ID {track_id} | Nose ({nose_x},{nose_y})"
            )

    # Resize window
    annotated_frame = cv2.resize(
        annotated_frame,
        (1200, 700)
    )

    cv2.imshow(
        "Pose + Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()

cv2.destroyAllWindows()

print("\nTrajectory data saved to trajectory.csv")
