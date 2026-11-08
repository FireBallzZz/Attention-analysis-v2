import cv2
from pathlib import Path

video_path = Path("C:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/runs/Test 5.mp4")
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0
cap.release()

print(f"Video: {video_path.name}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration:.2f} seconds")

