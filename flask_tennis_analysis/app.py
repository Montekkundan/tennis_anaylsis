from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import logging
import pandas as pd
import sys
import cv2
from copy import deepcopy
from moviepy.editor import VideoFileClip

# Add the project root to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import read_video, save_video, measure_distance, draw_player_stats, convert_pixel_distance_to_meters
from trackers import PlayerTracker, BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
from mini_court.mini_court import MiniCourt
import constants

app = Flask(__name__, static_folder="static")


# Configure logging
logging.basicConfig(level=logging.ERROR, filename='error_log.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up directories
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../input_videos')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'static/output_videos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def convert_to_mp4(input_path, output_path):
    """Convert .avi file to .mp4."""
    with VideoFileClip(input_path) as clip:
        clip.write_videofile(output_path, codec="libx264")

def process_video(video_path):
    """Process the uploaded video, using logic from main.py."""
    try:
        video_frames = read_video(video_path)
        player_tracker = PlayerTracker(model_path='../models/yolov8x.pt')
        ball_tracker = BallTracker(model_path='../models/last.pt')

        player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False)
        ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False)
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

        court_line_detector = CourtLineDetector('../models/keypoints_model.pth')
        court_keypoints = court_line_detector.predict(video_frames[0])

        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        mini_court = MiniCourt(video_frames[0])
        ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints)

        player_stats_data = [{
            'frame_num': 0,
            'player_1_number_of_shots': 0,
            'player_1_total_shot_speed': 0,
            'player_1_last_shot_speed': 0,
            'player_1_total_player_speed': 0,
            'player_1_last_player_speed': 0,
            'player_2_number_of_shots': 0,
            'player_2_total_shot_speed': 0,
            'player_2_last_shot_speed': 0,
            'player_2_total_player_speed': 0,
            'player_2_last_player_speed': 0,
        }]

        for ball_shot_ind in range(len(ball_shot_frames) - 1):
            start_frame = ball_shot_frames[ball_shot_ind]
            end_frame = ball_shot_frames[ball_shot_ind + 1]
            ball_shot_time_in_seconds = (end_frame - start_frame) / 24

            if start_frame in ball_mini_court_detections and end_frame in ball_mini_court_detections:
                distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                                   ball_mini_court_detections[end_frame][1])
                distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
                    distance_covered_by_ball_pixels,
                    constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court()
                )
                speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

                player_positions = player_mini_court_detections[start_frame]
                player_shot_ball = min(player_positions.keys(), 
                                       key=lambda player_id: measure_distance(player_positions[player_id],
                                                                              ball_mini_court_detections[start_frame][1]))

                opponent_player_id = 1 if player_shot_ball == 2 else 2
                if opponent_player_id in player_mini_court_detections[start_frame] and opponent_player_id in player_mini_court_detections[end_frame]:
                    distance_covered_by_opponent_pixels = measure_distance(
                        player_mini_court_detections[start_frame][opponent_player_id],
                        player_mini_court_detections[end_frame][opponent_player_id]
                    )
                    distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
                        distance_covered_by_opponent_pixels,
                        constants.DOUBLE_LINE_WIDTH,
                        mini_court.get_width_of_mini_court()
                    )
                    speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
                else:
                    speed_of_opponent = 0

                current_player_stats = deepcopy(player_stats_data[-1])
                current_player_stats['frame_num'] = start_frame
                current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
                current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
                current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
                current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
                current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

                player_stats_data.append(current_player_stats)

        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left').ffill()

        player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
        player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
        player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
        player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']

        output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
        output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))
        output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

        output_avi_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.splitext(os.path.basename(video_path))[0] + '.avi')
        save_video(output_video_frames, output_avi_path)

        output_mp4_path = output_avi_path.replace('.avi', '.mp4')
        convert_to_mp4(output_avi_path, output_mp4_path)

        return output_mp4_path

    except Exception as e:
        logging.error("Video processing failed", exc_info=True)
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)
            output_path = process_video(video_path)
            if output_path:
                return redirect(url_for('view_results', filename=os.path.basename(output_path)))
            else:
                return "Error processing video", 500
    return render_template('index.html')

@app.route('/results/<filename>')
def view_results(filename):
    return render_template('results.html', filename=filename)

@app.route('/static/output_videos/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
