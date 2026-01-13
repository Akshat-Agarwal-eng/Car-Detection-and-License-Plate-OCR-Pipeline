# Car-Detection-and-License-Plate-OCR-Pipeline@pdai

This project is a multi-stage pipeline implemented in a Jupyter Notebook. It processes an input video to detect cars, isolates individual cars, when they are near the camera, detects their license plates, and performs Optical Character Recognition (OCR) to read the plate number. The final recognized text is then validated against Indian license plate formats.

# Car Detection and License Plate OCR Pipeline

This project is a multi-stage pipeline implemented in a Jupyter Notebook. It processes an input video to detect cars, isolates individual cars when they are near the camera, detects their license plates, and performs Optical Character Recognition (OCR) to read the plate number. The final recognized text is then validated against Indian license plate formats.

## Project Pipeline

The code is broken down into three main steps, executed in sequence:

### Step 1: Car Detection in Video

* **File:** `car_detection&ocr.ipynb` (Cell 3)
* **Model:** YOLOv8 (`best (6).pt`) trained for 'Car' detection.
* **Input:** A source video file (e.g., `CarSev 3.mp4`).
* **Process:**
    1.  The `implement_detection` function loads the car detection YOLO model.
    2.  It reads the input video frame by frame.
    3.  On each frame, it runs the model to detect objects.
    4.  It filters these detections to isolate *only* the 'Car' class.
    5.  A green bounding box and a confidence label are drawn onto the frame for each detected car.
* **Output:** A new video file (`output_car_detection1.mp4`) showing the original video with all detected cars highlighted.

### Step 2: Car Isolation and Cropping

* **File:** `car_detection&ocr.ipynb` (Cell 4)
* **Input:** The video generated in Step 1 (`output_car_detection1.mp4`).
* **Process:**
    1.  The `process_video` function analyzes the new video. **Note:** Instead of re-running object detection, it uses OpenCV's color detection (`cv2.inRange`) to find the *green bounding boxes* drawn in Step 1.
    2.  It tracks these boxes using their centroids to identify movement.
    3.  It selects the largest car that is currently moving.
    4.  The `is_car_near` function checks if the selected car is "near" the camera based on its bounding box area, height, and distance from the bottom of the frame.
    5.  If a car is deemed "near," the `mask_car` function is called to create a new image containing *only* the car, with the rest of the frame blacked out.
* **Output:** A directory (`/content/masked_cars00`) filled with individual JPEG images of the isolated cars that passed the "nearness" check.

### Step 3: License Plate Detection & OCR

* **File:** `car_detection&ocr.ipynb` (Cell 5)
* **Models:**
    * YOLOv8 (`license-plate-finetune-v1l.pt`) for license plate detection.
    * Microsoft TrOCR (`microsoft/trocr-base-printed`) for Optical Character Recognition.
* **Input:** The directory of isolated car images (`/content/masked_cars00`).
* **Process:**
    1.  The script iterates through each car image from Step 2.
    2.  It uses the license plate YOLO model to find a plate (class ID 0) on the car.
    3.  Once a plate is found, it's cropped from the car image.
    4.  **Preprocessing:**
        * `remove_inr_sticker`: A small portion is cropped from the left side of the plate to remove the "IND" sticker.
        * `preprocess_plate_v2`: The plate is converted to grayscale, normalized, binarized using an adaptive threshold, and resized/padded to a standard 300x100 dimension.
    5.  **OCR:** The preprocessed plate image is passed to the TrOCR model (`trocr_predict`) to extract the text.
    6.  **Validation:** The recognized text is cleaned and validated by the `correct_plate_format` function, which checks it against standard Indian plate regex (`pattern_standard`), Bharat (BH) series regex (`pattern_bh`), and a set of `valid_state_codes`.
* **Output:** The script **does not print** the raw OCR text directly to the console. Instead, for *each valid* (Standard or BH) license plate, it displays a `matplotlib` plot. This plot shows:
    1.  The original car image.
    2.  The cropped plate.
    3.  The preprocessed plate image, with the **final validated OCR text as its title** (e.g., "OCR: MH20EE1999").
* Plates that are unread or fail validation are skipped and produce no output.

## Key Libraries (Dependencies)

To run this notebook, the following main libraries must be installed:

* `ultralytics`: For running the YOLOv8 models.
* `transformers`: For the TrOCR model.
* `torch`: Required by both `ultralytics` and `transformers`.
* `opencv-python` (cv2): For all video and image processing tasks.
* `matplotlib`: For displaying the final results.
* `Pillow` (PIL): Used for image handling before TrOCR.

## How to Run

1.  Ensure you have all dependencies installed (run Cell 1).
2.  Make sure the necessary files are in the correct paths (e.g., `/content/`):
    * **Models:** `best (6).pt`, `license-plate-finetune-v1l.pt`
    * **Input Video:** `CarSev 3.mp4`
3.  Run the notebook cells in sequential order.
