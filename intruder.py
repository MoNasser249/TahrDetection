import cv2
from ultralytics import YOLO
import numpy as np
import torch
import clip
from PIL import Image

# ----------------------------
# 1. Initialize Video Capture
# ----------------------------
VIDEO_PATH = "comb.mp4"  # Replace with your input video path
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Error opening video file: {VIDEO_PATH}")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ----------------------------
# 2. Initialize Video Writer
# ----------------------------
OUTPUT_VIDEO_PATH = "combobject_habitat_output.avi"  # Output video path
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# ----------------------------
# 3. Load Models
# ----------------------------
# 3.1 Load YOLO Object Detection Model
YOLO_WEIGHTS_PATH = "v11.pt"  # Ensure this path is correct
try:
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    # Print class mapping for verification
    print("YOLO Model Class Mapping:")
    for idx, name in yolo_model.names.items():
        print(f"Class ID {idx}: {name}")
except FileNotFoundError:
    raise FileNotFoundError(f"YOLO weights file not found at path: {YOLO_WEIGHTS_PATH}")
except Exception as e:
    raise Exception(f"Error loading YOLO model: {e}")

# 3.2 Load CLIP Model for Habitat Classification
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
except AttributeError as e:
    raise AttributeError("Failed to load CLIP model. Ensure OpenAI's CLIP is installed correctly.") from e
except Exception as e:
    raise Exception("An error occurred while loading the CLIP model.") from e

# 3.3 Define Habitat Labels and Names
# Define the textual descriptions for each habitat (used by CLIP)
habitat_descriptions = [
    "a vast desert with sand dunes",
    "a greenland scenery with green grass, river  and mountains",
    "a snowy environment with snow-covered trees"
]

# Define the corresponding habitat names
habitat_names = [
    "Desert",
    "Greenland",
    "Snow"
]

# Tokenize the habitat descriptions
text_tokens = clip.tokenize(habitat_descriptions).to(device)

# ----------------------------
# 4. Define Class Names and Colors
# ----------------------------
# Define class names according to the YOLO model's class ID assignments
class_names = {
    0: "Donkey",
    1: "Human",
    2: "Sheep",
    3: "Tahr"
}

# Define colors for bounding boxes and labels
donkey_bbox_color = (0, 255, 255)          # Cyan
human_bbox_color = (0, 0, 255)            # Red
sheep_bbox_color = (0, 255, 0)            # Green
tahr_bbox_color = (119, 0, 200)           # Purple

donkey_label_bg_color = (0, 255, 255)      # Cyan
human_label_bg_color = (0, 0, 255)        # Red
sheep_label_bg_color = (0, 255, 0)        # Green
tahr_label_bg_color = (119, 0, 200)       # Purple

intruder_bbox_color = (0, 0, 255)         # Red for intruders
intruder_label_bg_color = (0, 0, 255)     # Red for intruder labels

habitat_bg_color = (50, 50, 50)           # Dark Grey for habitat text background
text_color = (255, 255, 255)              # White

# ----------------------------
# 5. Define Utility Functions
# ----------------------------
def draw_text_with_background(
    frame, text, position, font_scale=0.5, font_thickness=1, bg_color=(0, 0, 255)
):
    """Draw text with a filled background rectangle."""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    x, y = position
    # Draw rectangle for background
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        -1
    )
    # Put text over the rectangle
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness
    )

def classify_habitat_clip(frame, model, text_tokens, habitat_names):
    """Classify the habitat of a frame using CLIP and return the habitat name."""
    # Convert the frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image for CLIP
    image_input = preprocess_clip(pil_image).unsqueeze(0).to(device)
    
    # Encode image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
                
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
                
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).squeeze(0)
        probabilities = similarity.softmax(dim=0).cpu().numpy()
    
    # Get the index of the highest probability
    habitat_idx = np.argmax(probabilities)
    
    # Retrieve the corresponding habitat name
    habitat = habitat_names[habitat_idx]
    
    return habitat

# ----------------------------
# 6. Process Video Frames
# ----------------------------
frame_count = 0  # To keep track of frame numbers (optional)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch the frame.")
        break
    
    frame_count += 1
    
    # 6.1 Object Detection using YOLO
    results = yolo_model(frame)
    detections = results[0].boxes  # Assuming YOLOv8 format
    
    # Initialize counts
    human_count = 0
    donkey_count = 0
    tahr_count = 0
    sheep_count = 0
    intruder_detected = False  # Flag to indicate if any intruder is detected
    
    # 6.2 Habitat Classification using CLIP
    habitat_label = classify_habitat_clip(frame, clip_model, text_tokens, habitat_names)
    
    # 6.3 Process Detections
    for detection in detections:
        # Extract bounding box coordinates, confidence, and class ID
        box = detection.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        confidence = detection.conf[0].cpu().item()
        class_id = int(detection.cls[0].cpu().item())
        
        # Debugging: Print detected class ID and class name
        detected_class_name = class_names.get(class_id, f"Class {class_id}")
        print(f"Frame {frame_count}: Detected {detected_class_name} with confidence {confidence:.2f}")
        
        # Filter out low-confidence detections
        if confidence < 0.2:
            continue
        
        # Get class name
        class_name = class_names.get(class_id, f"Class {class_id}")
        
        # Determine label and count based on class
        if class_id == 1:  # Human
            label = "Intruder-Human"
            color = human_bbox_color
            bg_color = human_label_bg_color
            human_count += 1
            intruder_detected = True
        elif class_id == 0:  # Donkey
            label = "Intruder-Donkey"
            color = donkey_bbox_color
            bg_color = donkey_label_bg_color
            donkey_count += 1
            intruder_detected = True
        elif class_id == 3:  # Tahr
            label = class_name
            color = tahr_bbox_color
            bg_color = tahr_label_bg_color
            tahr_count += 1
        elif class_id == 2:  # Sheep
            label = class_name
            color = sheep_bbox_color
            bg_color = sheep_label_bg_color
            sheep_count += 1
        else:
            label = class_name
            color = (255, 255, 255)  # White color for undefined classes
            bg_color = (0, 0, 0)      # Black background
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        draw_text_with_background(
            frame,
            label,
            (x1, y1 - 10),
            font_scale=0.5,
            font_thickness=1,
            bg_color=bg_color
        )
    
    # 6.4 Print Notifications for Intruders
    if intruder_detected:
        print(f"Frame {frame_count}: Intruder detected! Humans: {human_count}, Donkeys: {donkey_count}")
    
    # 6.5 Overlay Information on Frame
    y_offset = 40  # Starting y position
    
    # 6.5.1 Tahrs Count
    draw_text_with_background(
        frame,
        f"Tahrs: {tahr_count}",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=tahr_label_bg_color
    )
    
    # 6.5.2 Sheep Count
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Sheep: {sheep_count}",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=sheep_label_bg_color
    )
    
    # 6.5.3 Intruders Heading
    y_offset += 40
    draw_text_with_background(
        frame,
        "Intruders:",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=intruder_label_bg_color
    )
    
    # 6.5.4 Humans Count under Intruders
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Humans: {human_count}",
        (30, y_offset),  # Indented
        font_scale=0.7,
        font_thickness=2,
        bg_color=intruder_label_bg_color
    )
    
    # 6.5.5 Donkeys Count under Intruders
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Donkeys: {donkey_count}",
        (30, y_offset),  # Indented
        font_scale=0.7,
        font_thickness=2,
        bg_color=intruder_label_bg_color
    )
    
    # 6.5.6 Habitat Information
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Habitat: {habitat_label}",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=habitat_bg_color
    )
    
    # 6.6 Write the Frame to Output Video
    video_writer.write(frame)
    
    # 6.7 Optional: Display the Frame in Real-Time
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video processing interrupted by user.")
        break

# ----------------------------
# 7. Cleanup Resources and Save Output
# ----------------------------
# Release video capture and writer
cap.release()
video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
print("Video processing completed and output saved.")
