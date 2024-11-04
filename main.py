import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision
import cv2
from visualization_tools import draw_landmarks_on_image

model_path = 'pose_landmarker_full.task'

BaseOptions = tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

mp_drawing = mp.solutions.drawing_utils

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture('lvl_1.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp = int((frame_number / fps) * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp)
        print(pose_landmarker_result.pose_landmarks)

        frame_number += 1