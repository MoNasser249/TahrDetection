import cv2
from ultralytics import YOLO
import numpy as np
import torch
import clip
from PIL import Image
import os
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

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
YOLO_WEIGHTS_PATH = "V11NEW.pt"  # Ensure this path is correct
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
habitat_descriptions = [
    "a vast desert with sand dunes",
    "a greenland scenery with green grass, river and mountains",
    "a snowy environment with snow-covered trees"
]

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
class_names = {
    0: "Donkey",
    1: "Human",
    2: "Sheep",
    3: "Tahr"
}

# Define colors for bounding boxes and labels
colors = {
    "Donkey": (0, 255, 255),    # Cyan
    "Human": (0, 0, 255),       # Red
    "Sheep": (0, 255, 0),       # Green
    "Tahr": (119, 0, 200),      # Purple
    "Intruder-Human": (0, 0, 255),
    "Intruder-Donkey": (0, 0, 255),
    "Default": (255, 255, 255)
}

bg_colors = {
    "Donkey": (0, 255, 255),
    "Human": (0, 0, 255),
    "Sheep": (0, 255, 0),
    "Tahr": (119, 0, 200),
    "Intruder-Human": (0, 0, 255),
    "Intruder-Donkey": (0, 0, 255),
    "Default": (0, 0, 0),
    "Habitat": (50, 50, 50)
}

# ----------------------------
# 5. Define Paths for Low-Confidence Detections
# ----------------------------
LOW_CONF_DIR = "low_conf_detections"
LOW_CONF_IMAGES_DIR = os.path.join(LOW_CONF_DIR, "images")
LOW_CONF_LABELS_DIR = os.path.join(LOW_CONF_DIR, "labels")

# Create directories if they don't exist
os.makedirs(LOW_CONF_IMAGES_DIR, exist_ok=True)
os.makedirs(LOW_CONF_LABELS_DIR, exist_ok=True)

# ----------------------------
# 6. Define Utility Functions
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
        (255, 255, 255),
        font_thickness
    )

def classify_habitat_clip(frame, model, text_tokens, habitat_names):
    """Classify the habitat of a frame using CLIP and return the habitat name."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = preprocess_clip(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
                
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
                
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).squeeze(0)
        probabilities = similarity.softmax(dim=0).cpu().numpy()
    
    habitat_idx = np.argmax(probabilities)
    habitat = habitat_names[habitat_idx]
    
    return habitat

def save_low_confidence_detection(frame, detection, class_id, frame_count):
    """Save low-confidence detection images and labels."""
    box = detection.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    class_name = class_names.get(class_id, f"Class_{class_id}")
    confidence = detection.conf[0].cpu().item()
    
    # Ensure bounding box is within frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)
    
    # Extract the detected object region
    detected_obj = frame[y1:y2, x1:x2]
    
    if detected_obj.size == 0:
        return  # Skip if the bounding box is invalid
    
    # Define unique filenames
    timestamp = int(time.time())
    image_filename = f"frame{frame_count}_class{class_id}_{timestamp}.jpg"
    label_filename = f"frame{frame_count}_class{class_id}_{timestamp}.txt"
    
    # Save the image
    image_path = os.path.join(LOW_CONF_IMAGES_DIR, image_filename)
    cv2.imwrite(image_path, detected_obj)
    
    # Save the label in YOLO format (class_id x_center y_center width height)
    img_h, img_w, _ = frame.shape
    if img_w == 0 or img_h == 0:
        return  # Avoid division by zero
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    label_content = f"{class_id} {x_center} {y_center} {width} {height}\n"
    
    label_path = os.path.join(LOW_CONF_LABELS_DIR, label_filename)
    with open(label_path, "w") as f:
        f.write(label_content)

def retrain_yolo():
    """Retrain the YOLO model with low-confidence detections."""
    # Define paths
    TRAIN_DATA_DIR = "yolo_training_data"
    TRAIN_IMAGES_DIR = os.path.join(TRAIN_DATA_DIR, "images")
    TRAIN_LABELS_DIR = os.path.join(TRAIN_DATA_DIR, "labels")
    
    # Clear previous training data
    if os.path.exists(TRAIN_DATA_DIR):
        shutil.rmtree(TRAIN_DATA_DIR)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    
    # Copy low-confidence images and labels to training directories
    for filename in os.listdir(LOW_CONF_IMAGES_DIR):
        src = os.path.join(LOW_CONF_IMAGES_DIR, filename)
        dst = os.path.join(TRAIN_IMAGES_DIR, filename)
        shutil.copy(src, dst)
    
    for filename in os.listdir(LOW_CONF_LABELS_DIR):
        src = os.path.join(LOW_CONF_LABELS_DIR, filename)
        dst = os.path.join(TRAIN_LABELS_DIR, filename)
        shutil.copy(src, dst)
    
    # Prepare a YAML file for YOLO training
    yaml_content = f"""
train: {os.path.abspath(TRAIN_IMAGES_DIR)}
val: {os.path.abspath(TRAIN_IMAGES_DIR)}  # Using the same directory for validation
nc: {len(class_names)}
names: {list(class_names.values())}
"""
    yaml_path = os.path.join(TRAIN_DATA_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    # Initialize and train the model
    try:
        print("Starting YOLO retraining...")
        # Initialize a new YOLO model for training
        yolo_trainer = YOLO(YOLO_WEIGHTS_PATH)  # Load existing weights
        # Set the confidence threshold to 5% during training
        yolo_trainer.conf = 0.7  # This might vary based on the ultralytics version
        # Train the model
        training_result = yolo_trainer.train(data=yaml_path, epochs=1, imgsz=640)
        
        # Save the new best model
        new_weights_path = os.path.join(TRAIN_DATA_DIR, "best.pt")
        yolo_trainer.export(format="onnx")  # Example export; adjust as needed
        print(f"Retraining completed. New weights saved to {new_weights_path}")
        
        # Optionally, update the main YOLO model with the new weights
        # yolo_model = YOLO(new_weights_path)
        
    except Exception as e:
        print(f"Error during retraining: {e}")

def generate_confusion_matrix(y_true, y_pred, classes):
    """Generate and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Save the confusion matrix as an image
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

# ----------------------------
# 7. Process Video Frames
# ----------------------------
frame_count = 0  # To keep track of frame numbers

# Lists to store true and predicted labels for confusion matrix
# Note: In a real scenario, you need ground truth labels. Here, it's assumed that detections are predictions.
# This is a placeholder and should be replaced with actual labels if available.
y_true = []
y_pred = []

# Set YOLO model's confidence threshold to 5%
yolo_model.conf = 0.7  # Ensure this sets the detection threshold appropriately

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch the frame.")
        break
    
    frame_count += 1
    
    # Object Detection using YOLO
    results = yolo_model(frame)
    detections = results[0].boxes  # Assuming YOLOv8 format
    
    # Initialize counts
    human_count = 0
    donkey_count = 0
    tahr_count = 0
    sheep_count = 0
    intruder_detected = False  # Flag to indicate if any intruder is detected
    
    # Habitat Classification using CLIP
    habitat_label = classify_habitat_clip(frame, clip_model, text_tokens, habitat_names)
    
    # Process Detections
    for detection in detections:
        box = detection.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        confidence = detection.conf[0].cpu().item()
        class_id = int(detection.cls[0].cpu().item())
        
        detected_class_name = class_names.get(class_id, f"Class {class_id}")
        print(f"Frame {frame_count}: Detected {detected_class_name} with confidence {confidence:.2f}")
        
        # Collect predictions for confusion matrix
        y_pred.append(detected_class_name)
        # y_true should be collected from ground truth labels if available
        # y_true.append(actual_label)  # Placeholder
        
        # Filter out low-confidence detections for retraining (below 70%)
        if confidence < 0.7:
            save_low_confidence_detection(frame, detection, class_id, frame_count)
            # Optionally, skip further processing if you don't want to display these detections
            # continue  # Uncomment if you want to skip processing/displaying low-confidence detections
        
        # Only process and display detections with confidence >=70%
        if confidence < 0.7:
            continue  # Skip displaying low-confidence detections
        
        # Get class name
        class_name = class_names.get(class_id, f"Class {class_id}")
        
        # Determine label and count based on class
        if class_id == 1:  # Human
            label = "Intruder-Human"
            color = colors.get(label, colors["Default"])
            bg_color = bg_colors.get(label, bg_colors["Default"])
            human_count += 1
            intruder_detected = True
        elif class_id == 0:  # Donkey
            label = "Intruder-Donkey"
            color = colors.get(label, colors["Default"])
            bg_color = bg_colors.get(label, bg_colors["Default"])
            donkey_count += 1
            intruder_detected = True
        elif class_id == 3:  # Tahr
            label = class_name
            color = colors.get(label, colors["Default"])
            bg_color = bg_colors.get(label, bg_colors["Default"])
            tahr_count += 1
        elif class_id == 2:  # Sheep
            label = class_name
            color = colors.get(label, colors["Default"])
            bg_color = bg_colors.get(label, bg_colors["Default"])
            sheep_count += 1
        else:
            label = class_name
            color = colors["Default"]
            bg_color = bg_colors["Default"]
        
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
    
    # Print Notifications for Intruders
    if intruder_detected:
        print(f"Frame {frame_count}: Intruder detected! Humans: {human_count}, Donkeys: {donkey_count}")
    
    # Overlay Information on Frame
    y_offset = 40  # Starting y position
    
    # Tahrs Count
    draw_text_with_background(
        frame,
        f"Tahrs: {tahr_count}",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=bg_colors["Tahr"]
    )
    
    # Sheep Count
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Sheep: {sheep_count}",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=bg_colors["Sheep"]
    )
    
    # Intruders Heading
    y_offset += 40
    draw_text_with_background(
        frame,
        "Intruders:",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=bg_colors["Intruder-Human"]
    )
    
    # Humans Count under Intruders
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Humans: {human_count}",
        (30, y_offset),  # Indented
        font_scale=0.7,
        font_thickness=2,
        bg_color=bg_colors["Intruder-Human"]
    )
    
    # Donkeys Count under Intruders
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Donkeys: {donkey_count}",
        (30, y_offset),  # Indented
        font_scale=0.7,
        font_thickness=2,
        bg_color=bg_colors["Intruder-Donkey"]
    )
    
    # Habitat Information
    y_offset += 40
    draw_text_with_background(
        frame,
        f"Habitat: {habitat_label}",
        (10, y_offset),
        font_scale=0.7,
        font_thickness=2,
        bg_color=bg_colors["Habitat"]
    )
    
    # Write the Frame to Output Video
    video_writer.write(frame)
    
    # Optional: Display the Frame in Real-Time
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video processing interrupted by user.")
        break
    
    # Optional: Periodically retrain the model (e.g., every 100 frames)
    if frame_count % 100 == 0:
        retrain_yolo()

# ----------------------------
# 8. Generate Confusion Matrix
# ----------------------------
# Note: y_true should be populated with actual labels for a meaningful confusion matrix
# Here, it's assumed that detections are predictions without ground truth labels
# This is a placeholder and should be replaced with actual labels if available.

# Example placeholder (remove or replace with actual labels)
y_true = ["Human", "Sheep", "Tahr", "Donkey"]  # Replace with actual ground truth
y_pred = ["Human", "Sheep", "Tahr", "Donkey"]  # Replace with model predictions

# Uncomment and modify the following lines if ground truth labels are available
if y_true and y_pred:
     classes = list(class_names.values())
     generate_confusion_matrix(y_true, y_pred, classes)

# ----------------------------
# 9. Cleanup Resources and Save Output
# ----------------------------
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video processing completed and output saved.")
