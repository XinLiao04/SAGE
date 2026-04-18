import os
import cv2
import numpy as np
import pandas as pd

def sample_frames_uniform(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < num_frames:
        print(f"Insufficient frames: {video_path}")
        num_frames = frame_count

    interval = frame_count / num_frames

    frames = []
    next_idx = 0
    for i in range(frame_count):
        success, frame = cap.read()
        if not success:
            continue

        if i + 1 >= next_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            next_idx += interval

        if len(frames) == num_frames:
            break

    cap.release()
    return frames


def batch_sample_videos(video_dir, output_dir, num_frames=16):
    os.makedirs(output_dir, exist_ok=True)

    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = [
        f for f in os.listdir(video_dir)
        if any(f.lower().endswith(ext) for ext in video_exts)
    ]

    print(f"Find {len(video_files)} videos and start sampling...")

    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        frames = sample_frames_uniform(video_path, num_frames=num_frames)

        if len(frames) == 0:
            print(f"Skip {video_name} (cannot be read)")
            continue

        video_base = os.path.splitext(video_name)[-2]

        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_dir, f"{video_base}_{i + 1}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"{video_name}: Extracts {len(frames)} frames -> {frame_path}")


if __name__ == "__main__":
    video_dir = r"./video_dir"   # To Do
    frame_dir = r"./frame_dir"   # To Do
    n_frames = 16
    # extract from dir
    batch_sample_videos(video_dir, frame_dir, num_frames=n_frames)