import cv2

def read_video(video_path):
    # Check if the video path is valid
    if not video_path:
        raise ValueError("The video path is empty or invalid.")
    
    cap = cv2.VideoCapture(video_path)
    
    # Confirm that the video has been successfully opened
    if not cap.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # Check for empty frames to prevent errors in VideoWriter initialization
    if not output_video_frames:
        raise ValueError("No frames available to save.")
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
