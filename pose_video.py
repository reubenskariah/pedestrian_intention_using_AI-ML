from ultralytics import YOLO
import cv2

# Load YOLO Pose model
model = YOLO("yolov8n-pose.pt")

# Open video
cap = cv2.VideoCapture("walking people.mp4")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Pose estimation
    results = model(frame)

    # Print coordinates every 30th frame to avoid terminal flooding
    if frame_count % 30 == 0:

        print(f"\n========== Frame {frame_count} ==========")

        for person_idx, result in enumerate(results):

            if result.keypoints is not None:

                keypoints = result.keypoints.xy

                print(f"\nPerson {person_idx + 1}")

                for person in keypoints:

                    body_parts = [
                        "Nose",
                        "Left Eye",
                        "Right Eye",
                        "Left Ear",
                        "Right Ear",
                        "Left Shoulder",
                        "Right Shoulder",
                        "Left Elbow",
                        "Right Elbow",
                        "Left Wrist",
                        "Right Wrist",
                        "Left Hip",
                        "Right Hip",
                        "Left Knee",
                        "Right Knee",
                        "Left Ankle",
                        "Right Ankle"
                    ]

                    for i, point in enumerate(person):

                        x = int(point[0])
                        y = int(point[1])

                        print(f"{body_parts[i]:15} : ({x}, {y})")

    # Draw skeletons
    annotated_frame = results[0].plot()

    # Resize for display
    annotated_frame = cv2.resize(
        annotated_frame,
        (1200, 700)
    )

    cv2.imshow("Pose Coordinates", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()