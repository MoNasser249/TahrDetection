import cv2
from ultralytics import YOLO
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("comb.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Load YOLO model
model = YOLO("weightsv113/best.pt")  # Replace with the correct path to your model file

# Define minimum confidence threshold
MIN_CONFIDENCE = 0.5

# Correct class names dictionary based on data.yaml
class_names = {
    0: "Donkey",
    1: "Tahr",
    2: "human"
}

# Define colors
tahr_bbox_color = (119, 0, 200)  # Purple color for Tahr bounding boxes
tahr_label_bg_color = (119, 0, 200)  # Purple color for Tahr label background
intruder_bbox_color = (0, 0, 255)  # Red color for intruder bounding boxes
intruder_label_bg_color = (0, 0, 255)  # Red color for intruder label background
text_color = (255, 255, 255)  # White text color for all labels

def draw_text_with_background(frame, text, position, font_scale=0.5, font_thickness=1, bg_color=(0, 0, 255)):
    """Draw text with a filled background rectangle."""
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x, y = position
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

# Process video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processing completed or frame could not be read.")
        break

    # Detect objects in the frame
    results = model(frame)
    detections = results[0].boxes

    human_count = 0
    donkey_count = 0
    tahr_count = 0
    intruder_detected = False  # Reset the flag for each frame

    # Draw each detected object with the appropriate bounding box and label
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        confidence = detection.conf[0]
        class_id = int(detection.cls[0])

        # Ignore low-confidence detections
        if confidence < MIN_CONFIDENCE:
            continue

        class_name = class_names.get(class_id, f"Class {class_id}")
        label = f"{class_name}"

        if class_id == 2:  # human
            cv2.rectangle(frame, (x1, y1), (x2, y2), intruder_bbox_color, 2)
            draw_text_with_background(frame, label, (x1, y1 - 10), font_scale=0.5, font_thickness=1, bg_color=intruder_label_bg_color)
            human_count += 1
            intruder_detected = True  # Mark intruder detected
        elif class_id == 0:  # Donkey
            cv2.rectangle(frame, (x1, y1), (x2, y2), intruder_bbox_color, 2)
            draw_text_with_background(frame, label, (x1, y1 - 10), font_scale=0.5, font_thickness=1, bg_color=intruder_label_bg_color)
            donkey_count += 1
            intruder_detected = True  # Mark intruder detected
        elif class_id == 1:  # Tahr
            cv2.rectangle(frame, (x1, y1), (x2, y2), tahr_bbox_color, 2)
            draw_text_with_background(frame, label, (x1, y1 - 10), font_scale=0.5, font_thickness=1, bg_color=tahr_label_bg_color)
            tahr_count += 1

    # Print the notification for each frame where an intruder is detected
    if intruder_detected:
        print(f"Notification: Intruder detected! Human count: {human_count}, Donkey count: {donkey_count}")

    # Draw the counters on the original frame
    draw_text_with_background(frame, f"Tahrs: {tahr_count}", (10, 40), font_scale=0.7, font_thickness=2, bg_color=tahr_label_bg_color)
    draw_text_with_background(frame, f"Humans: {human_count}", (10, 100), font_scale=0.7, font_thickness=2, bg_color=intruder_label_bg_color)
    draw_text_with_background(frame, f"Donkeys: {donkey_count}", (10, 140), font_scale=0.7, font_thickness=2, bg_color=intruder_label_bg_color)

    # Write the frame with bounding boxes and counter overlay
    video_writer.write(frame)

    # Optional: Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
