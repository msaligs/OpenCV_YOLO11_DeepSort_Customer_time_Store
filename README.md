# Customer Time Tracking in Store
![Customer Tracking Demo](output/output_gif.gif)

## Overview
This project implements a real-time customer tracking system using computer vision to monitor how long customers spend in a store. It combines YOLO v11m for person detection with BoT-SORT tracking to maintain consistent customer IDs and measure their duration of stay.

## Getting Started
To get started with this project, follow these steps:

### Prerequisites
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/customer-time-tracking.git
    cd customer-time-tracking
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
1. Create a `config.yaml` file with your camera settings
2. Adjust detection parameters in `settings.py`
3. Place your video files in the `streams/` directory

### Running the System
1. For live camera feed:
    ```bash
    python time_notebook.py --source 0
    ```

2. For video file:
    ```bash
    python time_notebook.py --source streams/your_video.mp4
    ```


## Key Components
- **YOLO v11m**: Used for person detection
- **BoT-SORT**: Deep learning-based tracking algorithm
- **OpenCV**: For video processing and visualization
- **CUDA**: GPU acceleration support

## Features
- Real-time person detection
- Persistent ID tracking across frames
- Time measurement for each detected person
- Visual display of tracking information
- Video output capability

## Technical Implementation
1. **Initialization**
    - Sets up YOLO model with CUDA support if available
    - Initializes BoT-SORT tracker
    - Opens video source

2. **Main Processing Loop**
    - Reads video frames
    - Performs person detection using YOLO
    - Updates tracker with detection results
    - Maintains time dictionary for each tracked ID
    - Displays bounding boxes with IDs and time information

3. **Time Tracking**
    ```python
    if track_id not in tracker_time:
         tracker_time[track_id] = [perf_counter(), perf_counter()]
    tracker_time[track_id][1] = perf_counter()
    ```

## Requirements
- Python 3.x
- OpenCV
- Ultralytics YOLO
- CUDA-capable GPU (recommended)
- Deep SORT implementation
- Required Python packages:
  - opencv-python
  - ultralytics
  - torch
  - numpy
  - easydict

## Usage
1. Ensure all dependencies are installed
2. Place your input video in the streams directory
3. Run the script:
    ```bash
    python time_notebook.py
    ```

## Output
- Real-time visualization with:
  - Bounding boxes around detected persons
  - Tracking IDs
  - Time duration for each person
- Option to save processed video

## Limitations
- Requires good lighting conditions
- Performance depends on GPU capability
- May lose tracking in crowded scenes

## Future Improvements
- Add multiple camera support
- Implement zone-based analytics
- Add customer flow patterns analysis
- Export tracking data to database
