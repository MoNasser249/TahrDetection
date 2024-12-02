
# TahrDetection

![TahrDetection Logo](https://github.com/Dan-445)

![License](https://img.shields.io/github/license/Dan-445/TahrDetection)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-brightgreen)
![Ultralytics YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)
![CLIP](https://img.shields.io/badge/CLIP-OpenAI-lightgrey)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Retraining Mechanism](#retraining-mechanism)
- [Confusion Matrix](#confusion-matrix)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

TahrDetection is a **self-improving object detection and habitat classification pipeline** designed to detect Tahrs in video streams using advanced machine learning models. Combining the power of **YOLO (You Only Look Once)** for object detection and **CLIP (Contrastive Language-Image Pretraining)** for habitat classification, this pipeline processes video frames, identifies Tahrs, classifies their surrounding habitat, and iteratively enhances detection accuracy through automated retraining.

### Key Components

- **YOLO Object Detection**: Efficiently detects Tahrs within video frames with high accuracy and speed.
- **CLIP Habitat Classification**: Classifies the environmental habitat based on visual features extracted from video frames.
- **Self-Learning Mechanism**: Saves low-confidence detections for retraining, allowing the model to improve over time.
- **Confusion Matrix Generation**: Evaluates model performance by visualizing detection accuracy across different classes.

## Features

- **Real-Time Video Processing**: Processes video frames in real-time or batch mode.
- **High and Low Confidence Detection Handling**:
  - **Excludes** detections below *50% confidence** to reduce false positives.
  - **Saves** detections below **70% confidence** for retraining to enhance model accuracy.
- **Automated Retraining**: Periodically retrains the YOLO model using saved low-confidence detections.
- **Habitat Classification**: Utilizes CLIP to classify the habitat into predefined categories.
- **Visualization**: Overlays bounding boxes, labels, and habitat information on video frames.
- **Performance Monitoring**: Generates confusion matrices to monitor and evaluate model performance over time.

## Demo

![Demo Video](https://github.com/Dan-445/TahrDetection/raw/main/assets/demo.gif)

*Sample demonstration of the pipeline processing a video, detecting Tahrs, classifying habitat, and displaying annotations.*

## Requirements

Ensure you have the following installed:

- **Python 3.7 or higher**
- **pip** or **conda** package manager
- **GPU** with CUDA support (optional but recommended for faster processing)

### Python Libraries

- `opencv-python`
- `ultralytics`
- `numpy`
- `torch`
- `clip` (OpenAI's CLIP)
- `Pillow`
- `scikit-learn`
- `matplotlib`

## Installation

Follow these steps to set up the project environment:

### 1. Clone the Repository

```bash
git clone https://github.com/Dan-445/TahrDetection.git
cd TahrDetection
```

### 2. Create a Virtual Environment (Recommended)

Using `venv`:

```bash
python -m venv env
```

Activate the virtual environment:

- **Windows**:

  ```bash
  env\Scripts\activate
  ```

- **macOS/Linux**:

  ```bash
  source env/bin/activate
  ```

### 3. Install Dependencies

Using `pip`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*If a `requirements.txt` file is not provided, install the necessary packages manually:*

```bash
pip install opencv-python ultralytics numpy torch torchvision torchaudio clip scikit-learn matplotlib Pillow
```

### 4. Download YOLO Weights

Ensure you have the YOLO weights file (`v11.pt`). Place it in the project root directory or specify the correct path in the script.

You can download pre-trained YOLO models from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/yolov8).

## Usage

### 1. Prepare Your Video

Place your input video file (e.g., `comb.mp4`) in the project directory or specify its path in the script.

### 2. Configure the Script

Ensure the script parameters are set correctly:

- **VIDEO_PATH**: Path to the input video file.
- **YOLO_WEIGHTS_PATH**: Path to the YOLO weights file.
- **OUTPUT_VIDEO_PATH**: Desired path for the output video.

### 3. Run the Script

Execute the Python script:

```bash
python scripts/self_improving_yolo.py
```

*Replace `self_improving_yolo.py` with the actual script filename.*

### 4. Interact with the Script

- **Real-Time Display**: The script will display processed frames in a window. Press `q` to exit early.
- **Output Video**: The processed video with annotations will be saved to the specified `OUTPUT_VIDEO_PATH`.
- **Low-Confidence Detections**: Detections below 70% confidence are saved for retraining.
- **Retraining**: The model retrains automatically every 100 frames using the saved low-confidence detections.

## Directory Structure

```
TahrDetection/
│
├── low_conf_detections/
│   ├── images/
│   └── labels/
│
├── yolo_training_data/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── outputs/
│   ├── combobject_habitat_output.avi
│   ├── confusion_matrix.png
│   └── best.pt
│
├── scripts/
│   └── self_improving_yolo.py
│
├── assets/
│   ├── logo.png
│   └── demo.gif
│
├── requirements.txt
├── README.md
└── LICENSE
```

- **low_conf_detections/**: Stores images and labels of low-confidence detections for retraining.
- **yolo_training_data/**: Contains the training dataset and configuration for retraining YOLO.
- **outputs/**: Saves the output video, confusion matrix, and retrained model weights.
- **scripts/**: Contains the main Python script.
- **assets/**: Contains project logo and demo media.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Project documentation.
- **LICENSE**: License information.

## Configuration

You can adjust various parameters in the script to suit your needs:

- **Confidence Thresholds**:
  - **Detection Exclusion Threshold**: Set to exclude detections below 5% confidence.
  - **Retraining Threshold**: Save detections below 70% confidence for retraining.

- **Retraining Frequency**: Currently set to retrain every 100 frames. Adjust as needed.

- **Class Names and Colors**: Modify the `class_names`, `colors`, and `bg_colors` dictionaries to match your dataset and preferences.

- **Habitat Descriptions**: Update the `habitat_descriptions` and `habitat_names` lists to define the habitats relevant to your application.

## Retraining Mechanism

The script incorporates an automated retraining mechanism to enhance the YOLO model's accuracy over time:

1. **Detection Filtering**:
   - **Excludes** objects detected with confidence **below 5%** to minimize false positives.
   - **Saves** objects detected with confidence **below 70%** for retraining.

2. **Saving Low-Confidence Detections**:
   - Extracts and saves the bounding box region of low-confidence detections as images.
   - Saves corresponding labels in YOLO format (class ID and normalized bounding box coordinates).

3. **Periodic Retraining**:
   - Every 100 frames (configurable), the script initiates retraining.
   - Copies low-confidence detections into the training directory.
   - Generates a `data.yaml` configuration file required by YOLO.
   - Retrains the YOLO model for a specified number of epochs (e.g., 10 epochs).
   - Saves the new best model weights as `best.pt`.

4. **Model Update**:
   - Optionally, you can update the main YOLO model with the retrained weights to continue improving detection accuracy.

**Note**: Ensure that low-confidence detections are reviewed and correctly labeled before retraining to prevent introducing errors into the model.

## Confusion Matrix

A confusion matrix helps evaluate the performance of the object detection model by comparing predicted labels against actual labels.

### Generating the Confusion Matrix

1. **Collect Ground Truth Labels**: Populate the `y_true` list with actual object labels from the video frames.
2. **Collect Predictions**: The script already appends predicted labels to the `y_pred` list.
3. **Generate the Matrix**:
   
   ```python
   if y_true and y_pred:
       classes = list(class_names.values())
       generate_confusion_matrix(y_true, y_pred, classes)
   ```

   Ensure that both `y_true` and `y_pred` are populated accurately for this to work.

### Viewing the Confusion Matrix

The generated `confusion_matrix.png` will be saved in the `outputs/` directory. Open this image to visualize the model's performance across different classes.

## Troubleshooting

### Import Errors

- **Error**: `Import "sklearn.metrics" could not be resolved from sourcePylancereportMissingModuleSource`
  
  **Solution**: Ensure that `scikit-learn` is installed in your Python environment.

  ```bash
  pip install scikit-learn
  ```

  If using a virtual environment, ensure it's activated before installing.

### CUDA Errors

- **Error**: CUDA not available or GPU-related errors.
  
  **Solution**: Ensure that your system has a compatible NVIDIA GPU and the necessary CUDA drivers installed. Alternatively, you can run the script on CPU by ensuring that `device = "cpu"` in the script.

### Low-Confidence Detections Not Saving

- **Issue**: Low-confidence detections are not being saved as expected.
  
  **Solution**: Verify the confidence thresholds in the script and ensure that detections fall within the specified range (5% - 70%). Check the `low_conf_detections/images` and `low_conf_detections/labels` directories for saved files.

### Retraining Not Occurring

- **Issue**: The model is not retraining after the specified number of frames.
  
  **Solution**: Ensure that the `retrain_yolo()` function is being called correctly. Check for any errors in the retraining process by reviewing the console output. Also, verify that low-confidence detections exist for retraining.

## Contributing

Contributions are welcome! Follow these steps to contribute to the project:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add a detailed description of your changes"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

   Provide a clear description of your changes and the purpose behind them.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**: For the powerful YOLO object detection model.
- **[OpenAI CLIP](https://github.com/openai/CLIP)**: For the versatile CLIP model used in habitat classification.
- **[Scikit-learn](https://scikit-learn.org/)**: For providing robust tools for machine learning and evaluation.
- **[OpenCV](https://opencv.org/)**: For facilitating efficient video processing and computer vision tasks.
- **[Matplotlib](https://matplotlib.org/)**: For generating visualization plots like the confusion matrix.
- **Community Contributors**: Special thanks to all contributors who have helped improve this project.

---

*This project is a demonstration of integrating object detection with automated retraining mechanisms to create a self-improving model. It serves as a foundation for more complex and robust applications in computer vision and machine learning.*
```

---

## Instructions to Use the README

1. **Add the Logo and Demo Media**:
   
   - **Logo**:
     - Place your project logo image (`logo.png`) inside the `assets/` directory.
     - Ensure the path in the README (`assets/logo.png`) correctly points to the logo image in your repository.
   
   - **Demo GIF**:
     - Add a demo GIF (`demo.gif`) showcasing the project's functionality inside the `assets/` directory.
     - Update the path in the README (`assets/demo.gif`) if necessary.

2. **Create the `assets/` Directory** (if not already present):

   ```bash
   mkdir assets
   ```

3. **Add `requirements.txt`**:

   Create a `requirements.txt` file in the root directory with the following content:

   ```text
   opencv-python
   ultralytics
   numpy
   torch
   torchvision
   torchaudio
   clip
   scikit-learn
   matplotlib
   Pillow
   ```

   *This ensures that all necessary Python libraries are installed.*

4. **Update Paths and Links**:

   - **Repository Links**: Ensure that all URLs (e.g., logo, demo) correctly point to the files in your repository.
   - **Script Paths**: Verify that the script path (`scripts/self_improving_yolo.py`) matches your repository's structure.

5. **Commit and Push**:

   After adding and verifying all files, commit and push the changes to your GitHub repository.

   ```bash
   git add README.md assets/ requirements.txt
   git commit -m "Add comprehensive README.md with logo and demo"
   git push origin main
   ```

   *Replace `main` with your default branch name if different.*

6. **Verify on GitHub**:

   Navigate to your repository on GitHub to ensure that the `README.md` renders correctly, displaying the logo, badges, and all sections as intended.

## Additional Tips

- **Markdown Preview**: Use your code editor's Markdown preview feature (e.g., VS Code's Preview) to visualize the README and make formatting adjustments as needed.
  
- **Badges and Shields**: The README includes badges for license, Python version, PyTorch, YOLOv8, and CLIP. These badges enhance the visual appeal and provide quick insights into the project's key aspects.
  
- **Screenshots and Media**: Including a demo GIF or video helps users understand the project's functionality quickly.
  
- **Consistent Naming**: Ensure that class names and color schemes in the script match those described in the README for consistency.
  
- **License File**: Ensure that you include a `LICENSE` file in your repository if you mention the license in the README.
  
- **Contribution Guidelines**: Consider adding a `CONTRIBUTING.md` file with detailed guidelines if you expect external contributions.
