#!/usr/bin/env python3
import cv2
import os
import re
import sys
from pathlib import Path

def create_video_from_images(images_dir, output_video, fps=5):
    """
    Create a video from images in a directory with numeric sorting.
    
    Args:
        images_dir: Directory containing image files
        output_video: Output video file path
        fps: Frames per second (default: 5)
    """
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort numerically by extracting the number from filename
    def extract_number(filename):
        # Extract number from filename (e.g., "5.png" -> 5, "130.png" -> 130)
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    image_files.sort(key=extract_number)
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images. First few: {image_files[:5]}")
    print(f"Last few: {image_files[-5:]}")
    
    # Read first image to get dimensions
    first_image_path = os.path.join(images_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    
    if frame is None:
        print(f"Error: Could not read first image: {first_image_path}")
        return
    
    height, width, layers = frame.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write all frames
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not read image {image_file}, skipping...")
            continue
        
        video.write(img)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    video.release()
    print(f"Video created successfully: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_video_from_images.py <images_dir> [fps]")
        print("Example: python create_video_from_images.py /path/to/images 5")
        sys.exit(1)
    
    images_dir = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Output video in the same folder as images
    output_video = os.path.join(images_dir, "output_video.mp4")
    
    if not os.path.isdir(images_dir):
        print(f"Error: {images_dir} is not a valid directory")
        sys.exit(1)
    
    create_video_from_images(images_dir, output_video, fps)
