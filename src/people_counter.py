from ultralytics import YOLO
import cv2

model = YOLO('yolov8m.pt')
video_path = "input_videos/video5.mp4"
cap = cv2.VideoCapture(video_path)
line_y = 200

entry_count = 0 
exit_count = 0

previous_positions = {}
counted_ids = set()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    results = model.track(frame, persist=True, conf=0.3, classes=[0])
    annotated_frame = results[0].plot()
    
    cv2.line(annotated_frame, (0, line_y), (annotated_frame.shape[1], line_y), (0,255,255),2)
    cv2.putText(annotated_frame, "Counting Line:",(20,line_y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    boxes = results[0].boxes

    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for box, track_id in zip(xyxy, ids):
            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Mittelpunkt zeichnen
            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            if track_id in previous_positions:
                previous_y = previous_positions[track_id]

                if track_id not in counted_ids:
                    # Von oben nach unten
                    if previous_y < line_y and center_y >= line_y:
                        entry_count += 1
                        counted_ids.add(track_id)

                    # Von unten nach oben
                    elif previous_y > line_y and center_y <= line_y:
                        exit_count += 1
                        counted_ids.add(track_id)

            previous_positions[track_id] = center_y
    

    cv2.putText(annotated_frame, f"Entry: {entry_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Exit: {exit_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("People Counter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindwos()

print(f"Entry: ", entry_count)
print(f"Exit: ", exit_count)