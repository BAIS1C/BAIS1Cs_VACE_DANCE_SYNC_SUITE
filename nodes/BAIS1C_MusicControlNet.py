import torch
import numpy as np
import librosa
import json
import os
import glob
import cv2
from typing import List, Dict, Tuple, Optional

class BAIS1C_MusicControlNet:
    """
    BAIS1C VACE Dance Sync Suite - Music Control Net

    Professional music-to-pose synchronization node that:
    - Loads existing pose tensor JSONs with metadata
    - Applies frame-perfect BPM synchronization using industry formulas
    - Outputs synced pose tensors and optional stickman visualization
    - Saves synced results as new JSON files

    This is the core magic node for tempo-perfect dance animation sync.
    """

    @classmethod
    def _get_available_dances(cls) -> List[str]:
        try:
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

            if not suite_dir:
                return ["none"]

            library_dir = os.path.join(suite_dir, "dance_library")
            json_files = glob.glob(os.path.join(library_dir, "*.json"))

            dance_names = ["none"]
            for json_file in json_files:
                basename = os.path.basename(json_file)
                dance_name = os.path.splitext(basename)[0]
                dance_names.append(dance_name)

            return dance_names if len(dance_names) > 1 else ["none"]

        except Exception:
            return ["none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_bpm": ("FLOAT", {"default": 120.0, "min": 30.0, "max": 300.0}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 12.0, "max": 60.0}),
                "pose_source": (["library_json", "direct_tensor"], {"default": "direct_tensor"}),
                "input_pose_tensor": ("POSE", {
                    "visible_if": {"pose_source": "direct_tensor"}
                }),
                "library_dance": (cls._get_available_dances(), {
                    "visible_if": {"pose_source": "library_json"},
                    "default": "none"
                }),
                "sync_method": (["time_domain", "frame_perfect", "beat_aligned"], {"default": "frame_perfect"}),
                "loop_mode": (["once", "loop_to_fit", "crop_to_fit"], {"default": "loop_to_fit"}),
                "generate_video": ("BOOLEAN", {"default": True}),
                "video_style": (["stickman", "dots", "skeleton"], {"default": "stickman"}),
                "save_synced_json": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "synced_dance"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 896, "min": 256, "max": 2048}),
                "background": (["black", "white", "transparent"], {"default": "black"}),
                "smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("POSE", "IMAGE", "STRING")
    RETURN_NAMES = ("synced_pose_tensor", "pose_video", "sync_report")
    FUNCTION = "sync_pose_to_music"
    CATEGORY = "BAIS1C VACE Suite/Control"

    def sync_pose_to_music(self, audio, target_bpm, target_fps, pose_source, input_pose_tensor, library_dance,
                          sync_method, loop_mode, generate_video, video_style, save_synced_json,
                          output_filename, width, height, background, smoothing, debug):

        if debug:
            print(f"\n[BAIS1C Music Control Net] === SYNC STARTING ===")
            print(f"Target: {target_bpm:.1f} BPM @ {target_fps:.1f} FPS")
            print(f"Sync method: {sync_method}")

        # Step 1: Load pose data and extract source metadata
        pose_data, source_metadata = self._load_pose_data(
            pose_source, library_dance, input_pose_tensor, debug
        )

        if pose_data is None:
            error_msg = "Failed to load pose data"
            empty_tensor = torch.zeros((1, 128, 2), dtype=torch.float32)
            empty_video = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (empty_tensor, empty_video, error_msg)

        # Step 2: Calculate sync parameters using professional formulas
        sync_params = self._calculate_sync_parameters(
            source_metadata, target_bpm, target_fps, audio, sync_method, debug
        )

        # Step 3: Apply synchronization transformation
        synced_poses = self._apply_synchronization(
            pose_data, sync_params, loop_mode, smoothing, debug
        )

        # Step 4: Generate outputs
        sync_report = self._generate_sync_report(source_metadata, sync_params, synced_poses)

        pose_video = None
        if generate_video:
            pose_video = self._generate_pose_video(
                synced_poses, width, height, background, video_style, target_fps, debug
            )
        else:
            pose_video = torch.zeros((1, height, width, 3), dtype=torch.float32)

        # Step 5: Save synced JSON if requested
        if save_synced_json:
            self._save_synced_json(synced_poses, source_metadata, sync_params, output_filename, debug)

        if debug:
            print(f"[BAIS1C Music Control Net] === SYNC COMPLETE ===")
            print(f"Output: {synced_poses.shape[0]} frames @ {target_fps:.1f} FPS")

        return (synced_poses, pose_video, sync_report)

    def _load_pose_data(self, pose_source, library_dance, input_pose_tensor, debug):
        if pose_source == "direct_tensor":
            if input_pose_tensor is None:
                print(f"[BAIS1C Music Control Net] Error: No input pose tensor provided for direct_tensor")
                return None, None
            if debug:
                print(f"[BAIS1C Music Control Net] Using direct tensor input: {input_pose_tensor.shape}")
            metadata = {
                "source_bpm": 120.0,
                "source_fps": 24.0,
                "duration": input_pose_tensor.shape[0] / 24.0,
                "title": "direct_input"
            }
            if isinstance(input_pose_tensor, torch.Tensor):
                pose_data = input_pose_tensor.cpu().numpy()
            else:
                pose_data = np.array(input_pose_tensor)
            return pose_data, metadata

        elif pose_source == "library_json" and library_dance != "none":
            return self._load_from_json_library(library_dance, debug)
        else:
            print(f"[BAIS1C Music Control Net] Error: No valid pose source specified")
            return None, None

    def _load_from_json_library(self, library_dance, debug):
        try:
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

            if not suite_dir:
                raise FileNotFoundError("Dance library directory not found")

            json_path = os.path.join(suite_dir, "dance_library", f"{library_dance}.json")

            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Dance JSON not found: {json_path}")

            with open(json_path, 'r') as f:
                data = json.load(f)

            if "pose_tensor" not in data:
                raise ValueError("No pose_tensor found in JSON")

            pose_array = np.array(data["pose_tensor"], dtype=np.float32)

            metadata = {
                "source_bpm": data.get("metadata", {}).get("bpm", data.get("bpm", 120.0)),
                "source_fps": data.get("metadata", {}).get("fps", data.get("fps", 24.0)),
                "duration": data.get("metadata", {}).get("duration", data.get("duration", pose_array.shape[0] / 24.0)),
                "title": data.get("title", library_dance),
                "format": data.get("format_info", {}).get("format", "unknown"),
                "original_frames": pose_array.shape[0]
            }

            if debug:
                print(f"[BAIS1C Music Control Net] Loaded '{library_dance}': {pose_array.shape}")
                print(f"  Source: {metadata['source_bpm']:.1f} BPM @ {metadata['source_fps']:.1f} FPS")
                print(f"  Duration: {metadata['duration']:.2f}s ({metadata['original_frames']} frames)")

            return pose_array, metadata

        except Exception as e:
            print(f"[BAIS1C Music Control Net] Failed to load {library_dance}: {e}")
            return None, None

    def _calculate_sync_parameters(self, source_metadata, target_bpm, target_fps, audio, sync_method, debug):
        source_bpm = source_metadata["source_bpm"]
        source_fps = source_metadata["source_fps"]
        source_duration = source_metadata["duration"]
        original_frames = source_metadata.get("original_frames", None)
        if not original_frames:
            # Fallback for direct input
            original_frames = int(source_duration * source_fps)

        audio_duration = len(audio["waveform"]) / audio["sample_rate"]

        bpm_ratio = target_bpm / source_bpm
        fps_ratio = target_fps / source_fps

        if sync_method == "time_domain":
            stretch_factor = bpm_ratio
            target_frames = int(original_frames / stretch_factor)
        elif sync_method == "frame_perfect":
            source_frames_per_beat = (source_fps * 60) / source_bpm
            target_frames_per_beat = (target_fps * 60) / target_bpm
            frame_stretch_factor = target_frames_per_beat / source_frames_per_beat
            target_frames = int(original_frames * frame_stretch_factor)
        elif sync_method == "beat_aligned":
            total_beats_in_source = (source_duration * source_bpm) / 60
            target_duration_for_beats = (total_beats_in_source * 60) / target_bpm
            target_frames = int(target_duration_for_beats * target_fps)
        else:
            target_frames = original_frames

        target_duration = target_frames / target_fps

        sync_params = {
            "source_bpm": source_bpm,
            "source_fps": source_fps,
            "target_bpm": target_bpm,
            "target_fps": target_fps,
            "original_frames": original_frames,
            "target_frames": target_frames,
            "bpm_ratio": bpm_ratio,
            "fps_ratio": fps_ratio,
            "stretch_factor": target_frames / original_frames if original_frames else 1.0,
            "source_duration": source_duration,
            "target_duration": target_duration,
            "audio_duration": audio_duration,
            "sync_method": sync_method
        }

        if debug:
            print(f"[BAIS1C Music Control Net] Sync calculation:")
            print(f"  BPM: {source_bpm:.1f} → {target_bpm:.1f} (ratio: {bpm_ratio:.3f})")
            print(f"  FPS: {source_fps:.1f} → {target_fps:.1f} (ratio: {fps_ratio:.3f})")
            print(f"  Frames: {original_frames} → {target_frames} (factor: {sync_params['stretch_factor']:.3f})")
            print(f"  Duration: {source_duration:.2f}s → {target_duration:.2f}s")

        return sync_params

    def _apply_synchronization(self, pose_data, sync_params, loop_mode, smoothing, debug):
        original_frames = pose_data.shape[0]
        target_frames = sync_params["target_frames"]
        audio_duration = sync_params["audio_duration"]
        target_fps = sync_params["target_fps"]

        if loop_mode == "loop_to_fit":
            audio_frames_needed = int(audio_duration * target_fps)
            if target_frames < audio_frames_needed:
                loops_needed = int(np.ceil(audio_frames_needed / target_frames))
                extended_poses = np.tile(pose_data, (loops_needed, 1, 1))
                pose_data = extended_poses[:audio_frames_needed]
                target_frames = audio_frames_needed
        elif loop_mode == "crop_to_fit":
            audio_frames_needed = target_frames

        synced_poses = self._resample_pose_tensor(pose_data, target_frames)

        if smoothing > 0:
            synced_poses = self._apply_temporal_smoothing(synced_poses, smoothing)

        if debug:
            print(f"[BAIS1C Music Control Net] Synchronization applied:")
            print(f"  Resampled: {pose_data.shape[0]} → {synced_poses.shape[0]} frames")
            print(f"  Smoothing: {smoothing:.2f}")

        return torch.from_numpy(synced_poses).float()

    def _resample_pose_tensor(self, pose_tensor, target_frames):
        original_frames, num_points, coords = pose_tensor.shape

        if original_frames == target_frames:
            return pose_tensor

        old_indices = np.linspace(0, original_frames - 1, original_frames)
        new_indices = np.linspace(0, original_frames - 1, target_frames)

        resampled = np.zeros((target_frames, num_points, coords), dtype=np.float32)

        for point_idx in range(num_points):
            for coord_idx in range(coords):
                resampled[:, point_idx, coord_idx] = np.interp(
                    new_indices,
                    old_indices,
                    pose_tensor[:, point_idx, coord_idx]
                )

        return resampled

    def _apply_temporal_smoothing(self, poses, smoothing_factor):
        if smoothing_factor <= 0:
            return poses

        smoothed = poses.copy()

        for frame_idx in range(1, len(poses)):
            smoothed[frame_idx] = (
                smoothing_factor * smoothed[frame_idx - 1] +
                (1 - smoothing_factor) * poses[frame_idx]
            )

        return smoothed

    def _generate_sync_report(self, source_metadata, sync_params, synced_poses):
        timing_accuracy = abs(sync_params["target_duration"] - sync_params["audio_duration"])

        report = f"""BAIS1C Music Control Net - Sync Report
=====================================
Source: {source_metadata.get('title', 'Unknown')}
  Original: {sync_params['original_frames']} frames @ {sync_params['source_fps']:.1f} FPS
  BPM: {sync_params['source_bpm']:.1f}
  Duration: {sync_params['source_duration']:.2f}s

Target:
  Frames: {sync_params['target_frames']} @ {sync_params['target_fps']:.1f} FPS  
  BPM: {sync_params['target_bpm']:.1f}
  Duration: {sync_params['target_duration']:.2f}s

Synchronization:
  Method: {sync_params['sync_method']}
  Stretch Factor: {sync_params['stretch_factor']:.3f}
  BPM Ratio: {sync_params['bpm_ratio']:.3f}

Audio Match:
  Audio Duration: {sync_params['audio_duration']:.2f}s
  Timing Accuracy: {timing_accuracy:.3f}s difference

Result: {synced_poses.shape[0]} synchronized pose frames
Status: {'✅ SYNCED' if timing_accuracy < 0.1 else '⚠️ CHECK TIMING'}
"""

        return report

    def _generate_pose_video(self, poses, width, height, background, style, fps, debug):
        if isinstance(poses, torch.Tensor):
            poses_np = poses.cpu().numpy()
        else:
            poses_np = poses

        frames = []
        bg_color = {"black": (0, 0, 0), "white": (255, 255, 255), "transparent": (0, 0, 0)}[background]

        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (15, 17), (15, 18), (15, 19),
            (16, 20), (16, 21), (16, 22)
        ]

        for frame_idx, pose in enumerate(poses_np):
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            keypoints = []
            for x, y in pose[:23]:
                px = int(np.clip(x * width, 0, width - 1))
                py = int(np.clip(y * height, 0, height - 1))
                keypoints.append((px, py))

            if style == "stickman" or style == "skeleton":
                for i, j in connections:
                    if i < len(keypoints) and j < len(keypoints):
                        cv2.line(frame, keypoints[i], keypoints[j], (255, 255, 255), 2)
                for point in keypoints:
                    cv2.circle(frame, point, 4, (100, 200, 255), -1)
            elif style == "dots":
                for i, point in enumerate(keypoints):
                    color = (255, 100, 100) if i < 5 else (100, 255, 100)
                    cv2.circle(frame, point, 6, color, -1)

            if debug and frame_idx % 10 == 0:
                cv2.putText(frame, f"F:{frame_idx}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))

        if debug:
            print(f"[BAIS1C Music Control Net] Generated {len(frames)} video frames ({style} style)")

        return torch.stack(frames)

    def _save_synced_json(self, synced_poses, source_metadata, sync_params, filename, debug):
        try:
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

            if suite_dir:
                save_dir = os.path.join(suite_dir, "dance_library")
            else:
                save_dir = os.path.join(os.getcwd(), "output", "dance_library")

            os.makedirs(save_dir, exist_ok=True)

            if isinstance(synced_poses, torch.Tensor):
                pose_list = synced_poses.cpu().numpy().tolist()
            else:
                pose_list = synced_poses.tolist()

            synced_data = {
                "title": f"{filename}_synced",
                "source_title": source_metadata.get("title", "unknown"),
                "description": f"Synced version of {source_metadata.get('title', 'unknown')} using BAIS1C Music Control Net",
                "metadata": {
                    "bpm": sync_params["target_bpm"],
                    "fps": sync_params["target_fps"],
                    "duration": sync_params["target_duration"],
                    "frame_count": sync_params["target_frames"],
                    "sync_method": sync_params["sync_method"],
                    "source_bpm": sync_params["source_bpm"],
                    "source_fps": sync_params["source_fps"],
                    "stretch_factor": sync_params["stretch_factor"],
                    "sync_date": str(np.datetime64('now')),
                    "suite_version": "1.0.0"
                },
                "format_info": {
                    "format": source_metadata.get("format", "128-point"),
                    "shape": list(synced_poses.shape),
                    "coordinate_system": "normalized (0.0-1.0)"
                },
                "pose_tensor": pose_list
            }

            safe_filename = "".join(c for c in filename if c.isalnum() or c in ("-_")).strip()
            json_path = os.path.join(save_dir, f"{safe_filename}_synced.json")

            with open(json_path, 'w') as f:
                json.dump(synced_data, f, indent=2)

            if debug:
                print(f"[BAIS1C Music Control Net] Saved synced JSON: {json_path}")
                print(f"  File size: {os.path.getsize(json_path) / 1024:.1f} KB")

        except Exception as e:
            print(f"[BAIS1C Music Control Net] Failed to save JSON: {e}")

# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_MusicControlNet": BAIS1C_MusicControlNet}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_MusicControlNet": "🎵 BAIS1C Music Control Net"}
