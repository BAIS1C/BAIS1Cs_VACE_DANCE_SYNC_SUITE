import os
import json
import torch
import numpy as np
from tqdm import tqdm
import decord
from typing import Tuple, Dict, Any

# Import dwpose components
from ..dwpose.dwpose_detector import dwpose_detector


class BAIS1C_PoseExtractor:
    """
    Extract 128-point pose tensors from video using DWPose detection.
    Saves pose data as JSON files for use in dance sync workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),  # Video file path from video loader node
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
    RETURN_NAMES = ("pose_json_path", "pose_tensor")
    FUNCTION = "extract_poses"
    CATEGORY = "BASIC/Pose"
    OUTPUT_NODE = False

    def extract_poses(self, video: str, sample_stride: int, title: str, 
                     author: str = "", style: str = "", tempo: str = "", 
                     description: str = "", debug: bool = False) -> Tuple[str, torch.Tensor]:
        """
        Extract pose tensors from video and save to JSON file.
        
        Args:
            video: Path to video file
            sample_stride: Frame sampling rate (1=every frame, 2=every 2nd frame, etc.)
            title: Name for saved pose file
            debug: Enable debug output
            
        Returns:
            Tuple of (json_file_path, pose_tensor)
        """
        if debug:
            print(f"[BAIS1C PoseExtractor] Processing video: {video}")
            print(f"[BAIS1C PoseExtractor] Sample stride: {sample_stride}")

        try:
            # Extract pose data from video
            pose_tensor = self._extract_pose_tensor_from_video(video, sample_stride, debug)
            
            # Check for starter dance conflicts
            self._check_starter_conflict(title, debug)
            
            # Save to JSON file with metadata
            metadata = {
                "author": author,
                "style": style, 
                "tempo": tempo,
                "description": description
            }
            json_path = self._save_pose_json(pose_tensor, title, metadata, debug)
            
            if debug:
                print(f"[BAIS1C PoseExtractor] Saved pose data to: {json_path}")
                print(f"[BAIS1C PoseExtractor] Pose tensor shape: {pose_tensor.shape}")
            
            return (json_path, pose_tensor)
            
        except Exception as e:
            error_msg = f"[BAIS1C PoseExtractor] Error: {str(e)}"
            print(error_msg)
            # Return empty tensor and error path on failure
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

        # Load video using decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        
        # Adjust sample stride based on video FPS (similar to MimicMotion approach)
        fps = vr.get_avg_fps()
        adjusted_stride = sample_stride * max(1, int(fps / 24))
        
        # Get frame indices to process
        frame_indices = list(range(0, total_frames, adjusted_stride))
        
        if debug:
            print(f"[BAIS1C PoseExtractor] Video FPS: {fps:.2f}")
            print(f"[BAIS1C PoseExtractor] Total frames: {total_frames}")
            print(f"[BAIS1C PoseExtractor] Processing {len(frame_indices)} frames")
        
        # Extract frames and detect poses
        frames_batch = vr.get_batch(frame_indices).asnumpy()
        pose_sequence = []
        
        for i, frame in enumerate(tqdm(frames_batch, desc="Detecting poses", disable=not debug)):
            try:
                # Detect pose using DWPose
                pose_dict = dwpose_detector(frame)
                
                # Convert pose dict to 128-point tensor
                pose_tensor = self._pose_dict_to_tensor(pose_dict, frame.shape[:2])
                pose_sequence.append(pose_tensor)
                
            except Exception as e:
                if debug:
                    print(f"[BAIS1C PoseExtractor] Warning: Frame {i} pose detection failed: {e}")
                # Use previous pose or default pose on failure
                if pose_sequence:
                    pose_sequence.append(pose_sequence[-1].copy())
                else:
                    pose_sequence.append(self._create_default_pose())
        
        # Convert to tensor
        pose_array = np.stack(pose_sequence, axis=0)
        pose_tensor = torch.from_numpy(pose_array).float()
        
        return pose_tensor

    def _pose_dict_to_tensor(self, pose_dict: Dict[str, Any], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert DWPose detection results to 128-point tensor format.
        
        DWPose format:
        - bodies: [N, 18, 2] body keypoints  
        - faces: [N, 68, 2] face keypoints
        - hands: [42, 2] hand keypoints (21 per hand)
        
        Our 128-point format:
        - [0:18] = body keypoints (18 points)
        - [18:86] = face keypoints (68 points)  
        - [86:107] = left hand (21 points)
        - [107:128] = right hand (21 points)
        
        Args:
            pose_dict: DWPose detection results
            frame_shape: (height, width) of original frame
            
        Returns:
            Pose tensor of shape [128, 2] with normalized coordinates
        """
        height, width = frame_shape
        pose_tensor = np.zeros((128, 2), dtype=np.float32)
        
        try:
            # Extract body keypoints (0-17, total 18 points)
            bodies = pose_dict.get('bodies', {})
            if 'candidate' in bodies and len(bodies['candidate']) > 0:
                body_points = bodies['candidate'][:18]  # Take first 18 points
                # Body points are already normalized in DWPose
                pose_tensor[:len(body_points)] = body_points[:, :2]
            
            # Extract face keypoints (18-85, total 68 points)  
            faces = pose_dict.get('faces', np.array([]))
            if len(faces) > 0 and len(faces[0]) >= 68:
                face_points = faces[0][:68]  # Take first person, 68 face points
                # Face points are already normalized in DWPose
                pose_tensor[18:18+len(face_points)] = face_points[:, :2]
            
            # Extract hand keypoints
            hands = pose_dict.get('hands', np.array([]))
            if len(hands) >= 42:  # Should have 42 hand points (21 per hand)
                # Left hand (86-106, total 21 points)
                left_hand = hands[:21]
                pose_tensor[86:107] = left_hand[:, :2]
                
                # Right hand (107-127, total 21 points)  
                right_hand = hands[21:42]
                pose_tensor[107:128] = right_hand[:, :2]
                
        except Exception as e:
            print(f"[BAIS1C PoseExtractor] Warning: Pose conversion error: {e}")
            # Return default pose on conversion error
            return self._create_default_pose()
        
        return pose_tensor

    def _create_default_pose(self) -> np.ndarray:
        """
        Create a default T-pose when pose detection fails.
        
        Returns:
            Default pose tensor of shape [128, 2]
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
            x = face_center[0] + face_radius * np.cos(angle) 
            y = face_center[1] + face_radius * np.sin(angle)
            pose[18 + i] = [x, y]
        
        # Simple hand poses (21 points each)
        # Left hand
        left_wrist = pose[10]  # Left wrist position
        for i in range(21):
            offset_x = (i % 5) * 0.01 - 0.02
            offset_y = (i // 5) * 0.01
            pose[86 + i] = [left_wrist[0] + offset_x, left_wrist[1] + offset_y]
        
        # Right hand  
        right_wrist = pose[11]  # Right wrist position
        for i in range(21):
            offset_x = (i % 5) * 0.01 - 0.02
            offset_y = (i // 5) * 0.01
            pose[107 + i] = [right_wrist[0] + offset_x, right_wrist[1] + offset_y]
        
        return pose

    def _check_starter_conflict(self, title: str, debug: bool) -> None:
        """
        Check if title conflicts with bundled starter dances and warn user.
        
        Args:
            title: Proposed pose file title
            debug: Enable debug output
        """
        starter_dances = ["starter_hiphop", "starter_ballet", "starter_freestyle"]
        safe_title = "".join(c for c in title if c.isalnum() or c in ("-_")).rstrip()
        
        if safe_title.lower() in [s.lower() for s in starter_dances]:
            warning = f"[BAIS1C PoseExtractor] WARNING: '{title}' conflicts with bundled starter dance. Consider using a different name."
            print(warning)
            if debug:
                print(f"[BAIS1C PoseExtractor] Bundled starters: {starter_dances}")

    def _save_pose_json(self, pose_tensor: torch.Tensor, title: str, metadata: Dict[str, str], debug: bool) -> str:
        """
        Save pose tensor to JSON file with metadata.
        
        Args:
            pose_tensor: Pose data tensor [frames, 128, 2]
            title: Name for the pose file
            debug: Enable debug output
            
        Returns:
            Path to saved JSON file
        """
        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in ("-_")).rstrip()
        if not safe_title:
            safe_title = "untitled_pose"
        
        # Try to save in our dance_library folder first
        node_dir = os.path.dirname(os.path.abspath(__file__))
        suite_dir = os.path.dirname(node_dir)
        primary_save_dir = os.path.join(suite_dir, "dance_library")
        
        # Fallback to ComfyUI output folder
        fallback_save_dir = os.path.join(os.getcwd(), "output", "dance_library")
        
        for save_dir in [primary_save_dir, fallback_save_dir]:
            try:
                os.makedirs(save_dir, exist_ok=True)
                
                # Create JSON data with user metadata
                pose_data = {
                    "title": title,
                    "author": metadata.get("author", ""),
                    "style": metadata.get("style", ""),
                    "tempo": metadata.get("tempo", ""),
                    "description": metadata.get("description", "Pose tensor extracted using BAIS1C VACE Dance Sync Suite"),
                    "format": "128-point",
                    "shape": list(pose_tensor.shape),
                    "keypoint_structure": {
                        "body": "0-17 (18 points)",
                        "face": "18-85 (68 points)", 
                        "left_hand": "86-106 (21 points)",
                        "right_hand": "107-127 (21 points)"
                    },
                    "pose_tensor": pose_tensor.tolist()
                }
                
                # Save to file
                json_path = os.path.join(save_dir, f"{safe_title}.json")
                with open(json_path, 'w') as f:
                    json.dump(pose_data, f, indent=2)
                
                if debug:
                    print(f"[BAIS1C PoseExtractor] Successfully saved to: {json_path}")
                
                return json_path
                
            except (PermissionError, OSError) as e:
                if debug:
                    print(f"[BAIS1C PoseExtractor] Could not save to {save_dir}: {e}")
                continue
        
        # If both locations fail
        raise RuntimeError(f"Could not save pose file - no writable directory found")


# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🎯 Extract Pose Tensors (128pts)"}