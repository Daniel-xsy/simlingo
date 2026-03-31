#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CAMERA_LAYOUT = (
    ("rgb_front_left", "rgb_front", "rgb_front_right"),
    ("rgb_back_left", "rgb_back", "rgb_back_right"),
)
BEV_DIR = "bev"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an Orion multiview video from a route visualization directory. "
            "The output layout is a 2x3 camera grid on the left and BEV spanning "
            "both rows on the right."
        )
    )
    parser.add_argument("viz_dir", type=Path, help="Path to an Orion route viz directory")
    parser.add_argument("fps", nargs="?", type=int, default=5, help="Frames per second")
    return parser.parse_args()


def extract_frame_key(path: Path):
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.stem
    return float("inf"), path.stem


def list_images(image_dir: Path):
    return sorted(
        [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
        key=extract_frame_key,
    )


def load_frame_map(image_dir: Path):
    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No image files found in {image_dir}")
    return {path.stem: path for path in image_paths}


def validate_input_structure(viz_dir: Path):
    required_dirs = {name for row in CAMERA_LAYOUT for name in row}
    required_dirs.add(BEV_DIR)

    missing = [name for name in sorted(required_dirs) if not (viz_dir / name).is_dir()]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required subdirectories in {viz_dir}: {missing_list}")


def collect_frame_sets(viz_dir: Path):
    frame_maps = {}
    for row in CAMERA_LAYOUT:
        for view_name in row:
            frame_maps[view_name] = load_frame_map(viz_dir / view_name)
    frame_maps[BEV_DIR] = load_frame_map(viz_dir / BEV_DIR)

    reference_name = "rgb_front"
    reference_ids = set(frame_maps[reference_name].keys())
    mismatches = []

    for view_name, frame_map in frame_maps.items():
        frame_ids = set(frame_map.keys())
        if frame_ids != reference_ids:
            missing_here = sorted(reference_ids - frame_ids, key=lambda stem: extract_frame_key(Path(stem)))
            extra_here = sorted(frame_ids - reference_ids, key=lambda stem: extract_frame_key(Path(stem)))
            mismatch_parts = [f"{view_name}: expected {len(reference_ids)} frames, found {len(frame_ids)}"]
            if missing_here:
                mismatch_parts.append(f"missing {missing_here[:5]}")
            if extra_here:
                mismatch_parts.append(f"extra {extra_here[:5]}")
            mismatches.append("; ".join(mismatch_parts))

    if mismatches:
        raise ValueError("Frame ids are not aligned across views:\n" + "\n".join(mismatches))

    ordered_ids = sorted(reference_ids, key=lambda stem: extract_frame_key(Path(stem)))
    return ordered_ids, frame_maps


def read_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def resize_image(image, width, height):
    interpolation = cv2.INTER_AREA if image.shape[1] > width or image.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def determine_layout(frame_maps, first_frame_id):
    front_frame = read_image(frame_maps["rgb_front"][first_frame_id])
    bev_frame = read_image(frame_maps[BEV_DIR][first_frame_id])

    tile_height, tile_width = front_frame.shape[:2]
    total_height = tile_height * 2
    bev_height, bev_width = bev_frame.shape[:2]
    scaled_bev_width = max(1, int(round(bev_width * (total_height / bev_height))))

    return tile_width, tile_height, scaled_bev_width, total_height


def compose_frame(frame_id, frame_maps, tile_width, tile_height, bev_width, total_height):
    left_width = tile_width * 3
    canvas = np.zeros((total_height, left_width + bev_width, 3), dtype=np.uint8)

    for row_index, row in enumerate(CAMERA_LAYOUT):
        for col_index, view_name in enumerate(row):
            image = read_image(frame_maps[view_name][frame_id])
            resized = resize_image(image, tile_width, tile_height)
            y0 = row_index * tile_height
            x0 = col_index * tile_width
            canvas[y0:y0 + tile_height, x0:x0 + tile_width] = resized

    bev_image = read_image(frame_maps[BEV_DIR][frame_id])
    bev_resized = resize_image(bev_image, bev_width, total_height)
    canvas[:, left_width:left_width + bev_width] = bev_resized

    return canvas


def create_video(viz_dir: Path, fps: int):
    validate_input_structure(viz_dir)
    frame_ids, frame_maps = collect_frame_sets(viz_dir)

    if not frame_ids:
        raise ValueError(f"No synchronized frames found in {viz_dir}")

    tile_width, tile_height, bev_width, total_height = determine_layout(frame_maps, frame_ids[0])
    output_path = viz_dir / "output_video.mp4"
    output_size = (tile_width * 3 + bev_width, total_height)

    print(f"Found {len(frame_ids)} synchronized frames")
    print(f"Camera tile size: {tile_width}x{tile_height}")
    print(f"Output video size: {output_size[0]}x{output_size[1]}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for index, frame_id in enumerate(frame_ids, start=1):
            frame = compose_frame(frame_id, frame_maps, tile_width, tile_height, bev_width, total_height)
            video_writer.write(frame)
            if index % 10 == 0 or index == len(frame_ids):
                print(f"Processed {index}/{len(frame_ids)} frames...")
    finally:
        video_writer.release()

    print(f"Video created successfully: {output_path}")


def main():
    args = parse_args()
    if args.fps <= 0:
        print("Error: fps must be a positive integer", file=sys.stderr)
        return 1

    viz_dir = args.viz_dir
    if not viz_dir.is_dir():
        print(f"Error: {viz_dir} is not a valid directory", file=sys.stderr)
        return 1

    try:
        create_video(viz_dir, args.fps)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
