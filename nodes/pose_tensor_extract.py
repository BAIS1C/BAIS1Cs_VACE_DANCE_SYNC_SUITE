import os
import json
import torch
import numpy as np
from tqdm import tqdm
import decord
from typing import Tuple, Dict, Any

# Fix the import path - dwpose is a sibling directory to nodes
import sys
import inspect
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
    
    Extract 128-point pose tensors from video using DWPose detection.
    Saves pose data as JSON files for use in dance sync workflows.
    
    128-point format:
    - Body keypoints: 0-17 (18 points)
    - Face keypoints: 18-85 (68 points) 
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
        Returns tuple (metadata_summary_string, pose_tensor)
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
            pose_tensor = self._extract_pose_tensor_from_video(video_path, sample_stride, debug)
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

    def _extract_pose_tensor_from_video(self, video_path: str, sample_stride: int, debug: bool) -> torch.Tensor:
        """
        Extract pose data from video using DWPose detection.
        Args:
            video_path: Path to input video
            sample_stride: Frame sampling rate
            debug: Enable debug output
        Returns:
            Pose tensor of shape [frames, 128, 2]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            
            # Adjust stride based on video FPS to maintain consistent temporal sampling
            adjusted_stride = max(1, sample_stride * max(1, int(fps / 24)))
            frame_indices = list(range(0, total_frames, adjusted_stride))
            
            if debug:
                print(f"[BAIS1C VACE Suite] Video analysis:")
                print(f"  - Video FPS: {fps:.2f}")
                print(f"  - Total frames: {total_frames}")
                print(f"  - Sample stride: {sample_stride} -> {adjusted_stride} (FPS adjusted)")
                print(f"  - Processing {len(frame_indices)} frames")
                
            # Load frames in batch for efficiency
            frames_batch = vr.get_batch(frame_indices).asnumpy()
            
            pose_sequence = []
            failed_detections = 0
            
            # Process each frame with progress bar
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
            
        except Exception as e:
            print(f"[BAIS1C VACE Suite] Video processing failed: {e}")
            raise

    def _pose_dict_to_tensor(self, pose_dict: Dict[str, Any], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert DWPose detection results to 128-point tensor format.
        
        Format breakdown:
        0-17: Body keypoints (18 points)
        18-85: Face keypoints (68 points)
        86-106: Left hand keypoints (21 points)  
        107-127: Right hand keypoints (21 points)
        """
        height, width = frame_shape
        pose_tensor = np.zeros((128, 2), dtype=np.float32)
        
        try:
            # Extract body keypoints (0-17, total 18 points)
            bodies = pose_dict.get('bodies', {})
            if 'candidate' in bodies and len(bodies['candidate']) > 0:
                body_points = bodies['candidate'][:18]  # Take first 18 body points
                actual_points = min(len(body_points), 18)
                pose_tensor[:actual_points] = body_points[:actual_points, :2]
            
            # Extract face keypoints (18-85, total 68 points)
            faces = pose_dict.get('faces', np.array([]))
            if len(faces) > 0 and len(faces[0]) >= 68:
                face_points = faces[0][:68]  # Take first 68 face points
                pose_tensor[18:18+68] = face_points[:, :2]
            
            # Extract hand keypoints (86-127, total 42 points: 21 left + 21 right)
            hands = pose_dict.get('hands', np.array([]))
            if len(hands) >= 42:
                # Left hand: points 86-106 (21 points)
                left_hand = hands[:21]
                pose_tensor[86:107] = left_hand[:, :2]
                
                # Right hand: points 107-127 (21 points)
                right_hand = hands[21:42]
                pose_tensor[107:128] = right_hand[:, :2]
                
        except Exception as e:
            print(f"[BAIS1C VACE Suite] Warning: Pose conversion error: {e}")
            return self._create_default_pose()
            
        return pose_tensor

    def _create_default_pose(self) -> np.ndarray:
        """
        Create a default T-pose when pose detection fails.
        Returns: Default pose tensor of shape [128, 2] in normalized coordinates
        """
        pose = np.zeros((128, 2), dtype=np.float32)
        
        # Basic T-pose body keypoints (18 points)
        body_pose = np.array([
            [0.5, 0.1],   # 0: nose
            [0.5, 0.15],  # 1: neck
            [0.48, 0.12], # 2: right eye
            [0.52, 0.12], # 3: left eye
            [0.46, 0.14], # 4: right ear
            [0.54, 0.14], # 5: left ear
            [0.4, 0.25],  # 6: left shoulder
            [0.6, 0.25],  # 7: right shoulder
            [0.3, 0.4],   # 8: left elbow
            [0.7, 0.4],   # 9: right elbow
            [0.25, 0.55], # 10: left wrist
            [0.75, 0.55], # 11: right wrist
            [0.45, 0.6],  # 12: left hip
            [0.55, 0.6],  # 13: right hip
            [0.43, 0.8],  # 14: left knee
            [0.57, 0.8],  # 15: right knee
            [0.41, 1.0],  # 16: left ankle
            [0.59, 1.0],  # 17: right ankle
        ], dtype=np.float32)
        pose[:18] = body_pose
        
        # Simple face points centered on nose (68 points)
        face_center = np.array([0.5, 0.1])
        face_radius = 0.03
        for i in range(68):
            angle = 2 * np.pi * i / 68
            x