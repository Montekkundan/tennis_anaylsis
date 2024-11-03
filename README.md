# Tennis Analysis Project

This project analyzes tennis videos to detect and track the ball and players on the court. The analysis is performed using models trained with PyTorch and YOLO.

## Project Structure

```
.env
.gitattributes
.gitignore
analysis/
    ball_analysis.ipynb
constants/
    __init__.py
    __pycache__/
court_line_detector/
    __init__.py
    __pycache__/
    court_line_detector.py
error_log.log
input_videos/
main.py
flask_tennis_analysis/
mini_court/
    __init__.py
    __pycache__/
    mini_court.py
models/
    best.pt
    keypoints_model.pth
    last.pt
output_videos/
requirement.txt
runs/
    detect/
tracker_stubs/
    ball_detections.pkl
    player_detections.pkl
trackers/
    __init__.py
training/
utils/
venv/
yolo_inference.py
yolov8x.pt
```

## How It Works

1. **Input Videos**: Place your tennis match videos in the `input_videos/` directory.
2. **Run Analysis**: Execute the `main.py` script to start the analysis. This script will:
        - Detect court lines using the `court_line_detector` module.
        - Detect and track the ball and players using models from the `models/` directory.
3. **Output Videos**: The analyzed videos with detected and tracked objects will be saved in the `output_videos/` directory.

## Models

- **YOLO**: The `yolov8x.pt` model is used for object detection.
- **PyTorch**: The `best.pt`, `keypoints_model.pth`, and `last.pt` models are used for tracking and keypoint detection.

## Example

### Input Video

![Input Video](./input_videos/1.mp4)

### Output Video

![Output Video](./output_videos/test1.avi)

The output video shows the detected court lines, ball, and players with bounding boxes and keypoints.

## Requirements

Install the required dependencies using:

```sh
pip install -r requirement.txt
```

## Running the Analysis

To run the analysis, execute:

```sh
python main.py
```

## Notebooks

For detailed analysis and visualization, refer to the Jupyter notebooks in the `analysis/` directory, such as `ball_analysis.ipynb`.

and `training/` directory for training the models.

## License

This project is licensed under the MIT License.

This content provides a comprehensive overview of your project, including its structure, how it works, and how to run the analysis.