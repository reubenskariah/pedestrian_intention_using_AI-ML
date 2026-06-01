from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("people.jpg")

person_count = 0

for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])

        if class_id == 0:

            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(f"\nPerson {person_count}")
            print("Left   :", x1)
            print("Top    :", y1)
            print("Right  :", x2)
            print("Bottom :", y2)