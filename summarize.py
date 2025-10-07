# summarize.py
import os
import sys
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path

# Set thread limits for low-spec machines BEFORE importing numpy/cv2
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from tqdm import tqdm

def find_ffmpeg():
    """Check for ffmpeg executable and provide instructions if not found."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("ERROR: ffmpeg not found.", file=sys.stderr)
        print("Please install ffmpeg and ensure it's in your system's PATH.", file=sys.stderr)
        print("Installation guide: https://ffmpeg.org/download.html", file=sys.stderr)
        sys.exit(1)
    return ffmpeg_path

def run_ffmpeg(ffmpeg_path, command):
    """Execute an ffmpeg command, suppressing output."""
    process = subprocess.Popen(
        [ffmpeg_path] + command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.communicate()
    if process.returncode != 0:
        print(f"Warning: ffmpeg command failed with code {process.returncode}", file=sys.stderr)

def summarize_video(
    video_path: Path,
    output_path: Path,
    ratio: float,
    scale: int,
    fps: int,
    threshold: int,
    min_scene: int,
):
    """
    Generate a video summary using scene detection and ffmpeg.
    """
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}", file=sys.stderr)
        return

    ffmpeg_path = find_ffmpeg()
    
    # Use a temporary directory for intermediate clips
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Scene Detection using PySceneDetect
        print("Step 1: Detecting scenes...")
        video = open_video(str(video_path))
        
        # Downscale for performance
        if scale > 0:
            video.set_downscale_factor(video.frame_size[1] // scale)
            
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        # Process at a lower framerate
        scene_manager.detect_scenes(video, frame_skip=video.frame_rate.get_framerate() // fps - 1)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            print("No scenes detected. Cannot create summary.", file=sys.stderr)
            return
            
        print(f"Detected {len(scene_list)} scenes.")

        # 2. Select scenes to keep based on target ratio
        total_duration_frames = video.duration.get_frames()
        target_duration_frames = total_duration_frames * ratio
        
        # Sort scenes by duration (longer scenes are often more important)
        scene_list.sort(key=lambda s: s[1].get_frames() - s[0].get_frames(), reverse=True)
        
        selected_scenes = []
        current_duration_frames = 0
        for start_time, end_time in scene_list:
            scene_duration = end_time.get_frames() - start_time.get_frames()
            if scene_duration / video.frame_rate.get_framerate() < min_scene:
                continue
            if current_duration_frames < target_duration_frames:
                selected_scenes.append((start_time, end_time))
                current_duration_frames += scene_duration
            else:
                break
        
        # Sort back to chronological order for concatenation
        selected_scenes.sort(key=lambda s: s[0].get_frames())

        if not selected_scenes:
            print("No scenes met the minimum duration criteria.", file=sys.stderr)
            return

        # 3. Create clips using ffmpeg and write to a concat file
        print(f"Step 2: Extracting {len(selected_scenes)} summary clips...")
        concat_file_path = temp_path / "concat_list.txt"
        with open(concat_file_path, "w") as f:
            for i, (start, end) in enumerate(tqdm(selected_scenes, desc="Extracting clips")):
                clip_path = temp_path / f"clip_{i:04d}.mp4"
                command = [
                    "-y",
                    "-ss", str(start.get_seconds()),
                    "-to", str(end.get_seconds()),
                    "-i", str(video_path),
                    "-c", "copy",  # Fast stream copy
                    "-avoid_negative_ts", "1",
                    str(clip_path)
                ]
                run_ffmpeg(ffmpeg_path, command)
                f.write(f"file '{clip_path.resolve()}'\n")

        # 4. Concatenate clips into the final summary
        print("Step 3: Assembling final summary...")
        concat_command = [
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file_path),
            "-c", "copy",
            str(output_path)
        ]
        run_ffmpeg(ffmpeg_path, concat_command)
        
        final_duration = get_video_duration(output_path, ffmpeg_path)
        original_duration = video.duration.get_seconds()
        
        print("\n--- Summary Complete ---")
        print(f"Original Video: {video_path.name}")
        print(f"Summary Video:  {output_path.name}")
        print(f"Original Duration: {original_duration:.2f}s")
        print(f"Summary Duration:  {final_duration:.2f}s")
        print(f"Achieved Ratio:    {final_duration / original_duration:.2%}")
        print("------------------------")


def get_video_duration(video_path, ffmpeg_path):
    """Get video duration using ffprobe."""
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    command = [
        ffprobe_path, "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        return float(result.stdout)
    except (ValueError, IndexError):
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a fast video summary using scene detection.")
    parser.add_argument("input", type=Path, help="Path to the input video file.")
    parser.add_argument("-o", "--output", type=Path, default=Path("summary.mp4"), help="Path to the output summary file.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Target summary ratio (e.g., 0.1 for 10%).")
    parser.add_argument("--fps", type=int, default=2, help="Framerate to sample for analysis (1-5 fps is good).")
    parser.add_argument("--scale", type=int, default=360, help="Downscale frames to this height for faster analysis (e.g., 360). 0 to disable.")
    parser.add_argument("--threshold", type=int, default=27, help="PySceneDetect ContentDetector threshold (lower is more sensitive).")
    parser.add_argument("--min-scene", type=int, default=1, help="Minimum duration in seconds for a scene to be included.")
    
    args = parser.parse_args()
    
    # Set a sane number of threads for OpenCV
    try:
        cv2.setNumThreads(min(2, os.cpu_count() or 2))
    except:
        pass

    summarize_video(
        video_path=args.input,
        output_path=args.output,
        ratio=args.ratio,
        scale=args.scale,
        fps=args.fps,
        threshold=args.threshold,
        min_scene=args.min_scene,
    )