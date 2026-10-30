import os
import cv2
from pytubefix import YouTube, Playlist
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
import subprocess
import csv

def parse_playlist(source, output):
    """
    Parses a YouTube playlist URL and stores it to a csv file.
    
    Args:
        playlist_url (str): URL of the YouTube playlist
    """
    try:
        playlist = Playlist(source)

        with open(f'{output}/video.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for video in playlist.videos:
                writer.writerow([video.watch_url, video.title])

        print(f"Successfully wrote {len(playlist.videos)} Videos to {output}/video.csv")

    except Exception as e:
        print(f"An error occurred: {e}")


def download_youtube_video(
    youtube_url,
    output_dir,
    resolution='1080p',
    skip_intro=0,
    output_filename=None
):
    """
    Downloads a video from YouTube and removes the first X seconds (intro).
    
    Args:
        youtube_url (str): URL of the YouTube video
        output_dir (str): Directory where video will be saved
        resolution (str): Desired resolution of the video (e.g., '720p', '1080p')
        skip_intro (int): Number of seconds to skip from the beginning (0 to keep entire video) feature requires installed ffmpeg
        output_filename (str): Desired filename of the video
        
    Returns:
        str: Path to the downloaded and trimmed video file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download YouTube video
    print(f"Downloading video from: {youtube_url}")
    yt = YouTube(youtube_url)
    
    # Select stream with specified resolution
    stream = yt.streams.filter(file_extension='mp4', res=resolution).first()
    
    # If requested resolution is not available, get the highest resolution
    if not stream:
        print(f"Resolution {resolution} not available. Getting highest resolution.")
        stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
    
    # Set filename
    if output_filename is None:
        filename = stream.default_filename.replace(" ", "_")
    else:
        filename = output_filename
    
    # Set paths for original and trimmed videos
    original_video_path = output_path / f"original_{filename}"
    final_video_path = output_path / filename
    
    # Download the original video
    print(f"Downloading to: {original_video_path}")
    stream.download(output_path=str(output_path), filename=f"original_{filename}")
    
    if skip_intro > 0:
        print(f"Trimming: Removing first {skip_intro} seconds...")
        
        # Use FFmpeg to cut the first X seconds
        # -ss: start time, -c: codec (copy to avoid re-encoding), -avoid_negative_ts: handle negative timestamps
        ffmpeg_cmd = [
            'ffmpeg', '-i', str(original_video_path), 
            '-ss', str(skip_intro), 
            '-c', 'copy', 
            '-avoid_negative_ts', '1',
            str(final_video_path)
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Successfully trimmed video to: {final_video_path}")
            
            # Remove the original video
            os.remove(original_video_path)
        except subprocess.CalledProcessError as e:
            print(f"Error trimming video: {e}")
            print("Using original video instead.")
            os.rename(original_video_path, final_video_path)
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Using original video instead.")
            os.rename(original_video_path, final_video_path)
    else:
        # If no trimming needed, just rename the file
        os.rename(original_video_path, final_video_path)
    
    return str(final_video_path)


def extract_frames_from_video(
    video_path,
    output_dir,
    time_interval=1.0  # Extract frames every n seconds
):
    """
    Extracts frames from a video file at specified time intervals.
    Saves images in format [Video name]#t=[timestamp].jpg
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory where frames will be saved
        time_interval (float): Extract frames every n seconds
        
    Returns:
        str: Path to the directory containing extracted frames
    """
    # Create frames directory
    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video filename without extension to use in image names
    video_filename = Path(video_path).stem
    
    # Remove illegal characters from video filename
    video_filename = re.sub(r'[\\/*?:"<>|]', '_', video_filename)
    
    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Calculate frame interval from time interval
    frame_interval = max(1, int(fps * time_interval))  # Ensure at least 1 frame interval
    print(f"Using time interval of {time_interval} seconds (every {frame_interval} frames)")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Calculate current timestamp in seconds
            timestamp_seconds = frame_count / fps
            
            seconds = int(timestamp_seconds)
            milliseconds = int((timestamp_seconds - int(timestamp_seconds)) * 1000)
            
            timestamp_str = f"{seconds:02d}.{milliseconds:03d}"
            
            # Save the frame as an image with the format [Video name]#t=[timestamp].jpg
            frame_filename = frames_dir / f"{video_filename}#t={timestamp_str}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {saved_count} frames to {frames_dir}")
    return str(frames_dir)


def create_video_dataset(
    source,
    output,
    interval=1.0,
    resolution='1080p',
    output_filename=None
):
    """
    Universal function to create a dataset from a video.
    Detects if source is a YouTube URL or local file path and processes accordingly.
    
    Args:
        source (str): YouTube URL or path to local video file
        output (str): Directory where images will be saved
        interval (float): Extract frames every n seconds
        resolution (str): Desired resolution of the video (for YouTube downloads)
        output_filename (str): Desired filename of the video, only used if source is a YouTube URL
    Returns:
        str: Path to the directory containing extracted frames
    """
    # Ensure interval is a float
    try:
        interval = float(interval)
    except ValueError:
        print(f"Warning: Invalid interval value '{interval}'. Using default 1.0 seconds.")
        interval = 1.0
    
    # Create base output directory
    base_dir = Path(output)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect if source is a YouTube URL or local file path
    youtube_pattern = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'
    is_youtube = re.match(youtube_pattern, source)
    
    if is_youtube:
        # Source is a YouTube URL
        print("Detected YouTube URL")
        video_path = download_youtube_video(
            youtube_url=source,
            output_dir=output,
            resolution=resolution,
            skip_intro=30,  # Skip first 30 seconds of YouTube videos
            output_filename=output_filename
        )
    else:
        # Source is a local file path
        print("Detected local video file path")
        video_path = source
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Extract frames from the video
    extract_frames_from_video(
        video_path=video_path,
        output_dir=os.path.join(output, "images"),
        time_interval=interval
    )

    if is_youtube:
        # Delete downloaded video file if it was a YouTube download
        os.remove(video_path)