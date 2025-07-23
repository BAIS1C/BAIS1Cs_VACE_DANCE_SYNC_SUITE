import os
import json
import torch
import numpy as np
from tqdm import tqdm
import decord
from typing import Tuple, Dict, Any

import sys
import inspect

# Fix the import path for sibling directories
current_file = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, parent_dir)

try:
    from dwpose.dwpose_detector import dwpose_detector
except ImportError as e:
    print(f"[BAIS1C VACE Suite] Warning: Could not import DWPose detector: {e}")
    dwpose_detector = None

class BAIS1C_PoseExtractor:
    """
    BAIS1C VACE Dance Sync Suite - Pose Tensor Extractor

    Extracts 128-point pose tensors from video using DWPose detection.
    Saves pose data as JSON files for use in dance sync workflows.

    128-point format:
    - Body: 0-17 (18 points)
    - Face: 18-85 (68 points)
    - Left hand: 86-106 (21 points)
    - Right hand: 107-127 (21 points)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "bpm": ("FLOAT", {"default": 120.0, "min": 30.0, "max": 300.0}),
                "frame_count": ("INT", {"default": 240, "min": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 0.1}),
                "sample_stride": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "title": ("STRING", {"default": "extracted_poses"}),
            },
            "optional": {
                "author": ("STRING", {"default": ""}),
                "style": ("STRING", {"default": ""}),
                "tempo": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": ""}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "POSE")
    RETURN_NAMES = ("metadata_summary", "pose_tensor")
    FUNCTION = "extract_poses"
    CATEGORY = "BAIS1C VACE Suite/Extraction"
    OUTPUT_NODE = False

    def extract_poses(self, video, audio, fps, bpm, frame_count, duration, sample_stride, title, 
                     author="", style="", tempo="", description="", debug=False):
        """
        Extract pose tensors from video and save to JSON file.
        Returns: (metadata_summary_string, pose_tensor)
        """
        # Check if DWPose detector is available
        if dwpose_detector is None:
            error_msg = "[BAIS1C VACE Suite] Error: DWPose detector not initialized. Please check model files."
            print(error_msg)
            empty_tensor = torch.zeros((1, 128, 2), dtype=torch.float32)
            return (error_msg, empty_tensor)

        # Handle ComfyUI's VideoFromFile object or path string
        if hasattr(video, "video_path"):
            video_path = video.video_path
        elif isinstance(video, str):
            video_path = video
        else:
            raise ValueError(f"Unsupported video input type: {type(video)}")

        metadata_summary = (
            f"Title: {title}\n"
            f"FPS: {fps}\n"
            f"BPM: {bpm}\n"
            f"Duration: {duration:.2f}s\n"
            f"Frame count: {frame_count}\n"
            f"Audio sample rate: {audio['sample_rate']}\n"
            f"Sample stride: {sample_stride}"
        )

        if debug:
            print(f"\n[BAIS1C VACE Suite] Pose Extraction Starting:")
            print(metadata_summary)

        try:
            pose_tensor = self._extract_pose_tensor_from_video(video_path, fps, sample_stride, debug)
            self._check_starter_conflict(title, debug)

            # Save to JSON file with metadata
            metadata = {
                "author": author,
                "style": style,
                "tempo": tempo,
                "description": description,
                "fps": fps,
                "bpm": bpm,
                "duration": duration,
                "frame_count": frame_count,
                "sample_stride": sample_stride,
                "extraction_settings": {
                    "pose_format": "128-point",
                    "dwpose_version": "v1.0",
                    "stride_adjusted": True
                }
            }

            json_path = self._save_pose_json(pose_tensor, title, metadata, debug)

            if debug:
                print(f"[BAIS1C VACE Suite] ✅ Extraction complete!")
                print(f"  - Saved: {json_path}")
                print(f"  - Pose tensor shape: {pose_tensor.shape}")
                print(f"  - Points per frame: {pose_tensor.shape[1]}")
            return (metadata_summary, pose_tensor)

        except Exception as e:
            error_msg = f"[BAIS1C VACE Suite] Extraction failed: {str(e)}"
            print(error_msg)
            empty_tensor = torch.zeros((1, 128, 2), dtype=torch.float32)
            return (error_msg, empty_tensor)

    def _extract_pose_tensor_from_video(self, video_path: str, fps: float, sample_stride: int, debug: bool) -> torch.Tensor:
        """
        Extract pose data from video using DWPose detection.
        Args:
            video_path: Path to input video
            fps: target fps for stride adjustment
            sample_stride: user-supplied stride
            debug: Enable debug output
        Returns:
            Pose tensor of shape [frames, 128, 2]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

        # Adjust stride so the number of samples matches reference fps (default 24)
        adjusted_stride = max(1, int(sample_stride * max(1, int(round(video_fps / float(fps))))))
        frame_indices = list(range(0, total_frames, adjusted_stride))

        if debug:
            print(f"[BAIS1C VACE Suite] Video analysis:")
            print(f"  - Video FPS: {video_fps:.2f}")
            print(f"  - Total frames: {total_frames}")
            print(f"  - Adjusted stride: {sample_stride} -> {adjusted_stride} (FPS adjusted)")
            print(f"  - Processing {len(frame_indices)} frames")

        frames_batch = vr.get_batch(frame_indices).asnumpy()
        pose_sequence = []
        failed_detections = 0

        for i, frame in enumerate(tqdm(frames_batch, desc="Extracting poses", disable=not debug)):
            try:
                pose_dict = dwpose_detector(frame)
                pose_tensor = self._pose_dict_to_tensor(pose_dict, frame.shape[:2])
                pose_sequence.append(pose_tensor)
            except Exception as e:
                failed_detections += 1
                if debug:
                    print(f"[BAIS1C VACE Suite] Warning: Frame {i} detection failed: {e}")
                # Use previous pose or default pose
                if pose_sequence:
                    pose_sequence.append(pose_sequence[-1].copy())
                else:
                    pose_sequence.append(self._create_default_pose())

        if failed_detections > 0:
            print(f"[BAIS1C VACE Suite] Warning: {failed_detections}/{len(frame_indices)} pose detections failed")

        # Convert to tensor
        pose_array = np.stack(pose_sequence, axis=0)
        pose_tensor = torch.from_numpy(pose_array).float()

        if debug:
            print(f"[BAIS1C VACE Suite] Final pose tensor: {pose_tensor.shape}")
            print(f"  - Frames: {pose_tensor.shape[0]}")
            print(f"  - Points per frame: {pose_tensor.shape[1]}")
            print(f"  - Coordinates: {pose_tensor.shape[2]} (x, y)")

        return pose_tensor

    def _pose_dict_to_tensor(self, pose_dict: Dict[str, Any], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert DWPose detection results to 128-point tensor format.
        """
        height, width = frame_shape
        pose_tensor = np.zeros((128, 2), dtype=np.float32)

        try:
            # Body keypoints (0-17)
            bodies = pose_dict.get('bodies', {})
            if 'candidate' in bodies and len(bodies['candidate']) > 0:
                body_points = bodies['candidate'][:18]
                actual_points = min(len(body_points), 18)
                pose_tensor[:actual_points] = body_points[:actual_points, :2]

            # Face (18-85)
            faces = pose_dict.get('faces', np.array([]))
            if len(faces) > 0 and len(faces[0]) >= 68:
                face_points = faces[0][:68]
                pose_tensor[18:18+68] = face_points[:, :2]

            # Hands (86-127)
            hands = pose_dict.get('hands', np.array([]))
            if len(hands) >= 42:
                pose_tensor[86:107] = hands[:21, :2]
                pose_tensor[107:128] = hands[21:42, :2]

        except Exception as e:
            print(f"[BAIS1C VACE Suite] Warning: Pose conversion error: {e}")
            return self._create_default_pose()

        return pose_tensor

    def _create_default_pose(self) -> np.ndarray:
        pose = np.zeros((128, 2), dtype=np.float32)
        body_pose = np.array([
            [0.5, 0.1], [0.5, 0.15], [0.48, 0.12], [0.52, 0.12], [0.46, 0.14], [0.54, 0.14],
            [0.4, 0.25], [0.6, 0.25], [0.3, 0.4], [0.7, 0.4], [0.25, 0.55], [0.75, 0.55],
            [0.45, 0.6], [0.55, 0.6], [0.43, 0.8], [0.57, 0.8], [0.41, 1.0], [0.59, 1.0],
        ], dtype=np.float32)
        pose[:18] = body_pose

        # Face: circle points centered at nose
        face_center = np.array([0.5, 0.1])
        face_radius = 0.03
        for i in range(68):
            angle = 2 * np.pi * i / 68
            x = face_center[0] + face_radius * np.cos(angle)
            y = face_center[1] + face_radius * np.sin(angle)
            pose[18 + i] = [x, y]

        left_wrist = pose[10]
        right_wrist = pose[11]
        for i in range(21):
            offset_x = (i % 5) * 0.01 - 0.02
            offset_y = (i // 5) * 0.01
            pose[86 + i] = [left_wrist[0] + offset_x, left_wrist[1] + offset_y]
            pose[107 + i] = [right_wrist[0] + offset_x, right_wrist[1] + offset_y]
        return pose

    def _check_starter_conflict(self, title: str, debug: bool) -> None:
        starter_dances = ["starter_hiphop", "starter_ballet", "starter_freestyle"]
        safe_title = "".join(c for c in title if c.isalnum() or c in ("-_")).strip()
        if safe_title.lower() in [s.lower() for s in starter_dances]:
            warning = f"[BAIS1C VACE Suite] ⚠️ WARNING: '{title}' conflicts with bundled starter dance."
            print(warning)
            print(f"[BAIS1C VACE Suite] Consider using a different name to avoid confusion.")
            if debug:
                print(f"[BAIS1C VACE Suite] Bundled starters: {starter_dances}")

    def _save_pose_json(self, pose_tensor: torch.Tensor, title: str, metadata: Dict[str, Any], debug: bool) -> str:
        safe_title = "".join(c for c in title if c.isalnum() or c in ("-_")).strip()
        if not safe_title:
            safe_title = "untitled_pose"
        current_file = os.path.abspath(__file__)
        suite_dir = None
        check_dir = os.path.dirname(current_file)
        for _ in range(5):
            if os.path.exists(os.path.join(check_dir, 'dance_library')):
                suite_dir = check_dir
                break
            parent_dir = os.path.dirname(check_dir)
            if parent_dir == check_dir:
                break
            check_dir = parent_dir
        primary_save_dir = os.path.join(suite_dir, "dance_library") if suite_dir else None
        fallback_save_dir = os.path.join(os.getcwd(), "output", "dance_library")
        save_dirs = [d for d in [primary_save_dir, fallback_save_dir] if d is not None]
        for save_dir in save_dirs:
            try:
                os.makedirs(save_dir, exist_ok=True)
                pose_data = {
                    "title": title,
                    "author": metadata.get("author", ""),
                    "style": metadata.get("style", ""),
                    "tempo": metadata.get("tempo", ""),
                    "description": metadata.get("description", "Pose tensor extracted using BAIS1C VACE Dance Sync Suite"),
                    "metadata": {
                        "fps": metadata.get("fps", 0.0),
                        "bpm": metadata.get("bpm", 0.0),
                        "duration": metadata.get("duration", 0.0),
                        "frame_count": metadata.get("frame_count", 0),
                        "sample_stride": metadata.get("sample_stride", 1),
                        "extraction_date": str(np.datetime64('now')),
                        "suite_version": "1.0.0"
                    },
                    "format_info": {
                        "format": "128-point",
                        "shape": list(pose_tensor.shape),
                        "keypoint_structure": {
                            "body": "0-17 (18 points)",
                            "face": "18-85 (68 points)",
                            "left_hand": "86-106 (21 points)",
                            "right_hand": "107-127 (21 points)"
                        },
                        "coordinate_system": "normalized (0.0-1.0)",
                        "coordinate_order": "x, y"
                    },
                    "pose_tensor": pose_tensor.tolist()
                }
                json_path = os.path.join(save_dir, f"{safe_title}.json")
                with open(json_path, 'w') as f:
                    json.dump(pose_data, f, indent=2)
                if debug:
                    print(f"[BAIS1C VACE Suite] ✅ Pose data saved to: {json_path}")
                    print(f"[BAIS1C VACE Suite] File size: {os.path.getsize(json_path) / 1024:.1f} KB")
                return json_path
            except (PermissionError, OSError) as e:
                if debug:
                    print(f"[BAIS1C VACE Suite] Could not save to {save_dir}: {e}")
                continue
        raise RuntimeError(f"Could not save pose file - no writable directory found. Tried: {save_dirs}")

# Node registration (export at bottom)
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🎯 BAIS1C Pose Extractor (128pt)"}

# Self-test function (does not run on import)
def test_pose_extractor():
    """Test pose extractor functionality"""
    if dwpose_detector is None:
        print("❌ DWPose detector not available")
        return False

    extractor = BAIS1C_PoseExtractor()

    # Test default pose creation
    try:
        default_pose = extractor._create_default_pose()
        assert default_pose.shape == (128, 2)
        print("✅ Default pose creation test passed")
    except Exception as e:
        print(f"❌ Default pose creation failed: {e}")
        return False

    # Test pose dict conversion with dummy data
    try:
        dummy_pose_dict = {
            'bodies': {'candidate': np.random.rand(18, 3)},
            'faces': [np.random.rand(68, 3)],
            'hands': np.random.rand(42, 3)
        }
        converted_pose = extractor._pose_dict_to_tensor(dummy_pose_dict, (480, 640))
        assert converted_pose.shape == (128, 2)
        print("✅ Pose dict conversion test passed")
    except Exception as e:
        print(f"❌ Pose dict conversion failed: {e}")
        return False

    return True

# Uncomment the following to test:
# test_pose_extractor()
