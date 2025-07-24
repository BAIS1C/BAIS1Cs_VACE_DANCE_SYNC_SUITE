# music_control_net.py (BAIS1C VACE Dance Sync Suite – Auto-Sync Edition)
# --- FIXED VISUALIZATION ---
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
    BAIS1C VACE Dance Sync Suite – Music Control Net (Auto-Sync)
    • Auto-BPM from audio
    • Auto-length to match audio
    • Accepts direct pose tensor OR library JSON
    • Wan-safe aspect & resolution caps
    """

    # ---------- Wan-safe limits ----------
    MAX_W = 460
    MAX_H = 832
    VALID_RATIOS = {16/9, 9/16}

    @classmethod
    def INPUT_TYPES(cls):
        dances = cls._get_available_dances()
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 12.0, "max": 60.0}),

                "pose_source": (["direct_tensor", "library_json"], {"default": "direct_tensor"}),
                "library_dance": (dances, {"default": "none"}),

                "sync_method": (["time_domain", "frame_perfect", "beat_aligned"], {"default": "frame_perfect"}),
                "loop_mode": (["once", "loop_to_fit", "crop_to_fit"], {"default": "loop_to_fit"}),

                "generate_video": ("BOOLEAN", {"default": True}),
                "video_style": (["stickman", "dots", "skeleton"], {"default": "stickman"}),
                "save_synced_json": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "auto_synced"}),
                "smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_pose_tensor": ("POSE",),
            }
        }

    RETURN_TYPES = ("POSE", "IMAGE", "STRING")
    RETURN_NAMES = ("synced_pose_tensor", "pose_video", "sync_report")
    FUNCTION = "sync_pose_to_music"
    CATEGORY = "BAIS1C VACE Suite/Control"

    # ---------- Library scan ----------
    @classmethod
    def _get_available_dances(cls) -> List[str]:
        try:
            suite_dir = None
            check_dir = os.path.dirname(os.path.abspath(__file__))
            for _ in range(5):
                if os.path.exists(os.path.join(check_dir, "dance_library")):
                    suite_dir = check_dir
                    break
                parent = os.path.dirname(check_dir)
                if parent == check_dir:
                    break
                check_dir = parent
            if not suite_dir:
                return ["none"]
            lib = os.path.join(suite_dir, "dance_library")
            files = glob.glob(os.path.join(lib, "*.json"))
            names = ["none"] + \
                [os.path.splitext(os.path.basename(f))[0] for f in files]
            return names
        except Exception:
            return ["none"]

    # ---------- Auto-BPM ----------
    def _extract_bpm(self, audio: dict) -> float:
        try:
            # Use the new location for librosa.beat.tempo if available, fallback for older versions
            try:
                from librosa.feature.rhythm import tempo as librosa_tempo
            except ImportError:
                from librosa.beat import tempo as librosa_tempo

            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")

            # Ensure waveform is a numpy array on CPU
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.squeeze().cpu().numpy()
            else:
                waveform_np = np.array(waveform).squeeze(
                ) if waveform is not None else np.array([])

            # Handle stereo by converting to mono
            if waveform_np.ndim > 1:
                # Use -1 for last axis
                waveform_np = np.mean(waveform_np, axis=-1)

            # Check for empty waveform
            if waveform_np.size == 0:
                print(
                    "[BAIS1C MusicControlNet] Audio waveform is empty, cannot analyze BPM. Using default 120.0.")
                return 120.0

            # Normalize waveform
            if np.max(np.abs(waveform_np)) > 0:
                waveform_np = waveform_np / np.max(np.abs(waveform_np)) * 0.95

            tempo_estimates = librosa_tempo(
                y=waveform_np, sr=sample_rate, aggregate=None, start_bpm=60, std_bpm=40)
            bpm = float(tempo_estimates[0]) if len(
                tempo_estimates) > 0 else 120.0
            return bpm
        except Exception as e:
            print(
                f"[BAIS1C MusicControlNet] BPM extraction failed: {e}. Using default 120.0.")
            return 120.0

    # ---------- Wan-safe dimensions ----------
    def _clamp_res(self, w: int, h: int) -> Tuple[int, int]:
        # enforce 16:9 or 9:16
        ratio = w / h
        if abs(ratio - 16/9) < abs(ratio - 9/16):
            target_ratio = 16/9
        else:
            target_ratio = 9/16

        # scale to fit inside max box
        if w > h:  # landscape
            w_new = min(w, self.MAX_W)
            h_new = int(w_new / target_ratio)
            if h_new > self.MAX_H:
                h_new = self.MAX_H
                w_new = int(h_new * target_ratio)
        else:      # portrait
            h_new = min(h, self.MAX_H)
            w_new = int(h_new / target_ratio)
            if w_new > self.MAX_W:
                w_new = self.MAX_W
                h_new = int(w_new * target_ratio)

        return max(2, w_new), max(2, h_new)

    # ---------- Main sync ----------
    def sync_pose_to_music(self, audio, target_fps, pose_source, library_dance,
                           sync_method, loop_mode, generate_video, video_style,
                           save_synced_json, output_filename, smoothing, debug,
                           input_pose_tensor=None):

        if debug:
            print("\n[BAIS1C MusicControlNet] === Auto-Sync Start ===")

        # 1. Auto-BPM & length
        target_bpm = self._extract_bpm(audio)
        # --- Fix potential division by zero ---
        sample_rate = audio.get("sample_rate", 44100)
        if sample_rate <= 0:
            sample_rate = 44100
            print(
                "[BAIS1C MusicControlNet] Warning: Invalid sample rate, using default 44100.")
        waveform = audio.get("waveform")
        waveform_len = 0
        if isinstance(waveform, torch.Tensor):
            waveform_len = waveform.shape[-1]  # Assuming [C, T] or [T]
        elif waveform is not None:
            try:
                waveform_len = np.array(waveform).shape[-1]
            except:
                pass
        audio_duration = waveform_len / \
            sample_rate if sample_rate > 0 else 10.0  # Default fallback
        # --- Ensure positive target_fps ---
        target_fps = max(0.1, target_fps)
        # Ensure at least 1 frame
        target_frames = max(1, int(audio_duration * target_fps))

        if debug:
            print(f"[BAIS1C MusicControlNet] Target BPM: {target_bpm:.2f}")
            print(
                f"[BAIS1C MusicControlNet] Audio Duration: {audio_duration:.2f}s")
            print(f"[BAIS1C MusicControlNet] Target FPS: {target_fps:.2f}")
            print(
                f"[BAIS1C MusicControlNet] Target Frames (calculated): {target_frames}")

        # 2. Load pose
        pose_data_np = None
        source_metadata = {"source_bpm": 120,
                           "source_fps": 24, "title": "unknown"}
        if pose_source == "direct_tensor" and input_pose_tensor is not None:
            if isinstance(input_pose_tensor, torch.Tensor):
                pose_data_np = input_pose_tensor.cpu().numpy()
                source_metadata["title"] = "direct_input"
                # Infer source_fps if possible from target_frames and audio_duration?
                # For now, use defaults or values from sync_meta if it were passed.
                # Let's assume PoseExtractor provides good defaults in meta.
            else:
                print(
                    f"[BAIS1C MusicControlNet] Warning: input_pose_tensor is not a torch.Tensor. Type: {type(input_pose_tensor)}")
        elif pose_source == "library_json" and library_dance != "none":
            # Note: This path might need the old _load_from_json_library method
            # which is not present in this specific file version.
            # For now, we focus on direct_tensor path.
            # If library_json is used, implement _load_from_json_library or handle the error.
            try:
                # Placeholder for library loading logic if needed
                # pose_data_np, source_metadata = self._load_from_json_library(library_dance, debug)
                print(
                    "[BAIS1C MusicControlNet] Library JSON loading not implemented in this version for this node path.")
                pose_data_np = None  # This will trigger the error check below
            except Exception as e:
                print(
                    f"[BAIS1C MusicControlNet] Failed to load from library '{library_dance}': {e}")
                pose_data_np = None  # Will trigger the error check below

        # --- Critical Fix: Validate pose data before proceeding ---
        is_valid_pose = (
            pose_data_np is not None and
            isinstance(pose_data_np, np.ndarray) and
            pose_data_np.size > 0 and
            pose_data_np.ndim == 3 and
            pose_data_np.shape[1] == 128 and
            pose_data_np.shape[2] == 2
        )

        if not is_valid_pose:
            error_msg = f"[BAIS1C MusicControlNet] Error: Invalid or no pose data provided. " \
                f"Expected shape [F, 128, 2], got {getattr(pose_data_np, 'shape', 'None') if pose_data_np is not None else 'None'}."
            print(error_msg)
            # Return minimal valid outputs to prevent workflow crash
            dummy_pose = torch.zeros(
                (max(1, target_frames), 128, 2), dtype=torch.float32)
            # Ensure video tensor dimensions are valid
            dummy_w, dummy_h = self._clamp_res(
                512, 512)  # Use square defaults for dummy
            dummy_img = torch.zeros(
                (1, dummy_h, dummy_w, 3), dtype=torch.float32)
            return (dummy_pose, dummy_img, error_msg)

        original_pose_frames = pose_data_np.shape[0]
        if debug:
            print(f"[BAIS1C MusicControlNet] Loaded pose data:")
            print(f"  - Shape: {pose_data_np.shape}")
            print(f"  - Title: {source_metadata.get('title', 'N/A')}")

        # 3. Sync transform
        # --- Critical Fix: Pass validated data and add debug info ---
        synced_poses_tensor = self._resample_and_sync(
            pose_data_np, source_metadata, target_bpm, target_fps, sync_method, loop_mode, smoothing, target_frames, debug
        )
        # Ensure synced_poses_tensor is a tensor and has the correct last dimensions
        if not isinstance(synced_poses_tensor, torch.Tensor):
            synced_poses_tensor = torch.from_numpy(synced_poses_tensor).float()
        if synced_poses_tensor.ndim != 3 or synced_poses_tensor.shape[1] != 128 or synced_poses_tensor.shape[2] != 2:
            print(
                f"[BAIS1C MusicControlNet] Warning: Synced pose tensor has unexpected shape: {synced_poses_tensor.shape}")
            # Force correct shape for video generation, pad/crop if necessary in _generate_pose_video
            # For now, create a dummy tensor matching target frames
            synced_poses_tensor = torch.zeros(
                (max(1, target_frames), 128, 2), dtype=torch.float32)

        # 4. Wan-safe video
        w, h = self._clamp_res(512, 896)  # defaults
        pose_video_tensor = None
        if generate_video:
            # --- Critical Fix: Pass the potentially corrected tensor ---
            pose_video_tensor = self._generate_pose_video(
                synced_poses_tensor, w, h, "black", video_style, target_fps, debug)
        else:
            pose_video_tensor = torch.zeros(
                (max(1, synced_poses_tensor.shape[0]), h, w, 3), dtype=torch.float32)

        # 5. Optional save
        if save_synced_json:
            # Ensure synced_poses_tensor is on CPU for saving
            poses_for_saving = synced_poses_tensor.cpu()
            self._save_synced_json(
                poses_for_saving, source_metadata, target_bpm, target_fps, output_filename, debug)

        report = (f"Synced {source_metadata.get('title', 'unknown')} ({original_pose_frames} frames) → "
                  f"{synced_poses_tensor.shape[0]} frames @ {target_bpm:.1f} BPM, {target_fps:.1f} FPS")
        if debug:
            print(f"[BAIS1C MusicControlNet] === Auto-Sync Complete ===")
            print(f"[BAIS1C MusicControlNet] Report: {report}")
        return (synced_poses_tensor, pose_video_tensor, report)

    # ---------- Helpers ----------
    def _resample_and_sync(self, pose: np.ndarray, meta: dict, tgt_bpm: float, tgt_fps: float, method: str, loop: str, smooth: float, tgt_frames: int, debug: bool) -> torch.Tensor:
        """
        Resamples and synchronizes the pose data.
        Returns a PyTorch tensor of shape [target_frames, 128, 2].
        """
        # --- Critical Fix: Add robust input validation ---
        if pose is None or pose.size == 0:
            print(
                "[BAIS1C MusicControlNet] [_resample_and_sync] Error: Input pose array is empty.")
            return torch.zeros((max(1, tgt_frames), 128, 2), dtype=torch.float32)

        if pose.ndim != 3 or pose.shape[1] != 128 or pose.shape[2] != 2:
            print(
                f"[BAIS1C MusicControlNet] [_resample_and_sync] Error: Invalid input pose shape {pose.shape}. Expected [F, 128, 2].")
            return torch.zeros((max(1, tgt_frames), 128, 2), dtype=torch.float32)

        src_frames, pts, coords = pose.shape
        if src_frames <= 0:
            print(
                f"[BAIS1C MusicControlNet] [_resample_and_sync] Error: Source frame count is {src_frames}. Cannot process.")
            return torch.zeros((max(1, tgt_frames), pts, coords), dtype=torch.float32)

        # Ensure target frames is valid
        tgt_frames = max(1, tgt_frames)

        src_bpm = meta.get("source_bpm", 120)
        src_fps = meta.get("source_fps", 24)
        # Ensure positive values for ratios
        src_bpm = max(0.1, src_bpm)
        src_fps = max(0.1, src_fps)
        tgt_bpm = max(0.1, tgt_bpm)
        tgt_fps = max(0.1, tgt_fps)

        # ratios
        bpm_ratio = tgt_bpm / src_bpm
        fps_ratio = tgt_fps / src_fps

        if method == "frame_perfect":
            factor = (tgt_bpm / src_bpm) * (src_fps / tgt_fps)
        elif method == "beat_aligned":
            factor = (tgt_bpm / src_bpm)
        else:  # time_domain
            factor = tgt_bpm / src_bpm

        # Ensure factor is positive and reasonable
        factor = max(0.001, factor)
        new_len_float = src_frames * factor
        # Round and ensure at least 1
        new_len = max(1, int(round(new_len_float)))

        if debug:
            print(f"[BAIS1C MusicControlNet] [_resample_and_sync] Sync Details:")
            print(f"  - Source Frames: {src_frames}")
            print(f"  - Source BPM: {src_bpm:.2f}, FPS: {src_fps:.2f}")
            print(f"  - Target BPM: {tgt_bpm:.2f}, FPS: {tgt_fps:.2f}")
            print(f"  - Sync Method: {method}")
            print(
                f"  - BPM Ratio: {bpm_ratio:.3f}, FPS Ratio: {fps_ratio:.3f}")
            print(f"  - Stretch Factor: {factor:.3f}")
            print(f"  - Stretched Length (float): {new_len_float:.2f}")
            print(f"  - Stretched Length (int): {new_len}")
            print(f"  - Target Frames: {tgt_frames}")
            print(f"  - Loop Mode: {loop}")

        # handle loop/crop
        processed_pose = pose  # Start with original pose
        # --- Critical Fix: Ensure processed_pose has valid frames before tiling/cropping ---
        if processed_pose.shape[0] <= 0:
            print(
                "[BAIS1C MusicControlNet] [_resample_and_sync] Error: Processed pose has no frames before loop/crop.")
            return torch.zeros((tgt_frames, pts, coords), dtype=torch.float32)

        if loop == "loop_to_fit":
            # Calculate how many loops are needed to reach at least tgt_frames
            if new_len > 0:  # Prevent division by zero
                loops_needed = int(np.ceil(tgt_frames / new_len))
                if loops_needed > 0:
                    # Tile the data
                    processed_pose = np.tile(
                        processed_pose, (loops_needed, 1, 1))
                # Ensure we don't exceed tgt_frames unnecessarily, but slicing will handle that
                # Slice to the exact target length needed
                processed_pose = processed_pose[:tgt_frames]
            else:
                # If new_len is somehow 0 or negative, fallback
                print(
                    f"[BAIS1C MusicControlNet] [_resample_and_sync] Warning: new_len is {new_len} in loop_to_fit. Using original pose.")
                # Use original pose, potentially repeated once
                processed_pose = np.tile(processed_pose, (int(
                    np.ceil(tgt_frames / src_frames)), 1, 1))[:tgt_frames]

        elif loop == "crop_to_fit":
            # Determine the number of frames to take
            # It should be the minimum of the stretched length and the target
            frames_to_take = min(new_len, tgt_frames)
            # Ensure we don't try to take more frames than we have
            frames_to_take = min(frames_to_take, processed_pose.shape[0])
            if frames_to_take > 0:
                processed_pose = processed_pose[:frames_to_take]
            else:
                # If frames_to_take is 0, take at least 1 frame
                print(
                    f"[BAIS1C MusicControlNet] [_resample_and_sync] Warning: frames_to_take is {frames_to_take} in crop_to_fit. Taking 1 frame.")
                processed_pose = processed_pose[:1]
            # Update tgt_frames for resampling if cropping changed the target
            # No, tgt_frames is fixed by audio duration. We resample *to* tgt_frames.

        # --- Critical Fix: Final check before resampling ---
        final_frames_before_resample = processed_pose.shape[0]
        if final_frames_before_resample <= 0:
            print(
                f"[BAIS1C MusicControlNet] [_resample_and_sync] Error: No frames available for resampling after {loop}. Frame count: {final_frames_before_resample}")
            return torch.zeros((tgt_frames, pts, coords), dtype=torch.float32)

        # --- Critical Fix: Only resample if frame count differs ---
        if final_frames_before_resample == tgt_frames:
            if debug:
                print(
                    f"[BAIS1C MusicControlNet] [_resample_and_sync] Frame counts match ({tgt_frames}), skipping resampling.")
            synced = processed_pose
        else:
            # --- Critical Fix: Ensure indices are created correctly ---
            # Create interpolation indices
            # np.linspace(start, stop, num) where num > 0
            old_idx = np.linspace(
                0, final_frames_before_resample - 1, final_frames_before_resample)
            new_idx = np.linspace(
                0, final_frames_before_resample - 1, tgt_frames)

            # --- Critical Fix: Check for empty arrays before interpolation ---
            if old_idx.size == 0:
                print(
                    f"[BAIS1C MusicControlNet] [_resample_and_sync] Error: old_idx is empty. final_frames_before_resample={final_frames_before_resample}")
                return torch.zeros((tgt_frames, pts, coords), dtype=torch.float32)

            if new_idx.size == 0:
                print(
                    f"[BAIS1C MusicControlNet] [_resample_and_sync] Error: new_idx is empty. tgt_frames={tgt_frames}")
                return torch.zeros((tgt_frames, pts, coords), dtype=torch.float32)

            if debug:
                print(f"[BAIS1C MusicControlNet] [_resample_and_sync] Resampling:")
                print(f"  - Old indices shape: {old_idx.shape}")
                print(f"  - New indices shape: {new_idx.shape}")
                print(f"  - Processed pose shape: {processed_pose.shape}")

            # resample
            synced = np.zeros((tgt_frames, pts, coords), dtype=np.float32)
            try:
                for i in range(pts):
                    for c in range(coords):
                        # Perform linear interpolation
                        synced[:, i, c] = np.interp(
                            new_idx, old_idx, processed_pose[:, i, c])
            except Exception as e:
                print(
                    f"[BAIS1C MusicControlNet] [_resample_and_sync] Interpolation failed: {e}")
                return torch.zeros((tgt_frames, pts, coords), dtype=torch.float32)

        # --- Critical Fix: Ensure synced array is the correct shape ---
        if synced.shape[0] != tgt_frames:
            print(
                f"[BAIS1C MusicControlNet] [_resample_and_sync] Warning: Synced array has {synced.shape[0]} frames, expected {tgt_frames}. Padding/Cropping.")
            # Pad or crop to match exactly
            if synced.shape[0] < tgt_frames:
                # Pad with the last frame
                padding_needed = tgt_frames - synced.shape[0]
                last_frame = synced[-1:] if synced.shape[0] > 0 else np.zeros(
                    (1, pts, coords), dtype=np.float32)
                padding_frames = np.tile(last_frame, (padding_needed, 1, 1))
                synced = np.concatenate([synced, padding_frames], axis=0)
            else:
                # Crop
                synced = synced[:tgt_frames]

        # smooth
        if smooth > 0:
            # Convert to float64 for numerical stability during smoothing if needed
            # synced = synced.astype(np.float64)
            for t in range(1, len(synced)):
                smoothed_point = smooth * \
                    synced[t-1] + (1 - smooth) * synced[t]
                # Handle potential NaNs or Infs from smoothing
                if not np.all(np.isfinite(smoothed_point)):
                    # If smoothing produces invalid values, skip smoothing for this point
                    # or use the original point
                    synced[t] = synced[t]  # Keep original
                    if debug and t < 5:  # Log only first few instances
                        print(
                            f"[BAIS1C MusicControlNet] [_resample_and_sync] Warning: Smoothing produced NaN/Inf at frame {t}. Keeping original.")
                else:
                    synced[t] = smoothed_point
            # Convert back to float32 if it was changed
            # if synced.dtype != np.float32:
            #     synced = synced.astype(np.float32)

        if debug:
            print(
                f"[BAIS1C MusicControlNet] [_resample_and_sync] Final synced pose shape: {synced.shape}")

        return torch.from_numpy(synced).float()

    # ---------- FIXED Visualization ----------
    def _generate_pose_video(self, poses_tensor: torch.Tensor, w: int, h: int, bg: str, style: str, fps: float, debug: bool) -> torch.Tensor:
        """Generate pose visualization as image tensor - FIXED VERSION"""
        try:
            # --- Robust Input Handling ---
            # Ensure poses_tensor is a numpy array on CPU
            if isinstance(poses_tensor, torch.Tensor):
                # Detach in case of gradients, then move to CPU and convert
                poses_np = poses_tensor.detach().cpu().numpy()
            else:
                # If it's somehow already a numpy array (less likely from torch input)
                poses_np = np.array(poses_tensor)

            # Validate shape rigorously
            if poses_np.ndim != 3:
                raise ValueError(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] Expected 3D pose tensor, got {poses_np.ndim}D with shape {poses_np.shape}")
            if poses_np.shape[1] != 128 or poses_np.shape[2] != 2:
                raise ValueError(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] Expected pose shape [F, 128, 2], got {poses_np.shape}")

            num_frames = poses_np.shape[0]
            if num_frames == 0:
                print(
                    "[BAIS1C MusicControlNet] [_generate_pose_video] Warning: No frames in pose data. Generating single blank frame.")
                return torch.zeros((1, h, w, 3), dtype=torch.float32)

            if debug:
                # Check coordinate ranges for the first frame to diagnose issues
                first_pose = poses_np[0]
                print(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] DEBUG: First frame X range: [{np.min(first_pose[:, 0]):.3f}, {np.max(first_pose[:, 0]):.3f}]")
                print(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] DEBUG: First frame Y range: [{np.min(first_pose[:, 1]):.3f}, {np.max(first_pose[:, 1]):.3f}]")
                # Check a few specific points
                print(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] DEBUG: First frame Nose (0): ({first_pose[0, 0]:.3f}, {first_pose[0, 1]:.3f})")
                print(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] DEBUG: First frame RShoulder (6): ({first_pose[6, 0]:.3f}, {first_pose[6, 1]:.3f})")
                print(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] DEBUG: First frame LKnee (13): ({first_pose[13, 0]:.3f}, {first_pose[13, 1]:.3f})")

            frames = []
            # --- Background Color ---
            bg_color_map = {"black": (0, 0, 0), "white": (
                255, 255, 255), "transparent": (0, 0, 0)}  # Added transparent
            # Default to black if color not found or invalid
            bg_color_bgr = bg_color_map.get(bg.lower(), (0, 0, 0))

            # --- Pose Drawing Configuration ---
            # Define body connections for stickman/skeleton (using first 17 body points)
            # Indices: 0-17 are typically body keypoints in COCO/DWPose order
            body_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            # Indices for head (0-4), body (5-10), legs (11-16) for different colors/styles if needed
            # For dots, we can just draw all 128 points or a subset

            # --- Frame Generation Loop ---
            for frame_idx, pose in enumerate(poses_np):
                # Create blank frame
                frame_bgr = np.full((h, w, 3), bg_color_bgr, dtype=np.uint8)

                # --- Critical Fix: Coordinate Conversion and Validation ---
                # 1. Clamp normalized coordinates to [0, 1] to prevent out-of-bounds issues
                clamped_pose = np.clip(pose, 0.0, 1.0)
                # 2. Convert normalized coordinates (0-1) to pixel coordinates (0-width/height)
                # Using w-1 and h-1 to stay within image bounds
                keypoints_px = np.round(
                    clamped_pose * [w - 1, h - 1]).astype(int)
                # keypoints_px is now an array of [x_px, y_px] integer pixel coordinates

                # --- Drawing Based on Style ---
                if style in ("stickman", "skeleton"):
                    # Draw lines for body connections
                    for i, j in body_connections:
                        # Get pixel coordinates for the two joints
                        pt1 = tuple(keypoints_px[i])
                        pt2 = tuple(keypoints_px[j])
                        # Draw line if both points are valid (they should be after clipping)
                        # Add a check to ensure points are within image bounds (redundant but safe)
                        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                            cv2.line(frame_bgr, pt1, pt2,
                                     (255, 255, 255), 2)  # White lines

                    # Draw circles for joints (all 128 points or just body?)
                    # Let's draw all 128 for maximum information
                    for i, (px, py) in enumerate(keypoints_px):
                        # Check bounds (redundant after clipping and rounding, but safe)
                        if 0 <= px < w and 0 <= py < h:
                            # Different colors for different body parts if desired
                            if 0 <= i <= 4:  # Head (0-4)
                                color = (0, 0, 255)  # Red
                            elif 5 <= i <= 10:  # Arms/Shoulders (5-10)
                                color = (0, 255, 0)  # Green
                            elif 11 <= i <= 16:  # Legs/Hips (11-16)
                                color = (255, 0, 0)  # Blue
                            elif 17 <= i <= 85:  # Face (17-85)
                                color = (255, 255, 0)  # Cyan
                            elif 86 <= i <= 106:  # Left Hand (86-106)
                                color = (255, 0, 255)  # Magenta
                            else:  # Right Hand (107-127)
                                color = (0, 255, 255)  # Yellow
                            cv2.circle(frame_bgr, (px, py), 3,
                                       color, -1)  # Filled circle

                elif style == "dots":
                    # Draw only joint points (all 128)
                    for i, (px, py) in enumerate(keypoints_px):
                        if 0 <= px < w and 0 <= py < h:
                            # Different colors for different body parts
                            if 0 <= i <= 4:  # Head
                                color = (0, 0, 255)  # Red
                            elif 5 <= i <= 10:  # Arms/Shoulders
                                color = (0, 255, 0)  # Green
                            elif 11 <= i <= 16:  # Legs/Hips
                                color = (255, 0, 0)  # Blue
                            elif 17 <= i <= 85:  # Face
                                color = (255, 255, 0)  # Cyan
                            elif 86 <= i <= 106:  # Left Hand
                                color = (255, 0, 255)  # Magenta
                            else:  # Right Hand
                                color = (0, 255, 255)  # Yellow
                            # Slightly larger filled circle
                            cv2.circle(frame_bgr, (px, py), 4, color, -1)

                # Add frame number for debugging (Fixed logic: show on more frames if debug is on)
                if debug:  # Show frame number on ALL frames if debug is enabled
                    cv2.putText(frame_bgr, f"F:{frame_idx}", (10, 30),
                                # Green text
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- Convert BGR frame to RGB tensor format ---
                # ComfyUI IMAGE format is [H, W, C] with float values 0-1
                # cv2 creates BGR, so we need to convert to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb.astype(
                    np.float32) / 255.0)  # Normalize to 0-1
                frames.append(frame_tensor)

            # --- Stack frames into a batch tensor ---
            if frames:
                # Resulting shape: [Num_Frames, Height, Width, Channels]
                pose_viz_tensor = torch.stack(frames, dim=0)
            else:
                # Shouldn't happen due to checks, but safe fallback
                print(
                    "[BAIS1C MusicControlNet] [_generate_pose_video] Warning: No frames generated. Returning blank tensor.")
                pose_viz_tensor = torch.zeros(
                    (1, h, w, 3), dtype=torch.float32)

            if debug:
                print(
                    f"[BAIS1C MusicControlNet] [_generate_pose_video] Generated video tensor shape: {pose_viz_tensor.shape}")

            return pose_viz_tensor

        except Exception as e:
            print(
                f"[BAIS1C MusicControlNet] [_generate_pose_video] Failed to generate pose video: {e}")
            import traceback
            traceback.print_exc()
            # Return a blank video tensor on failure
            return torch.zeros((1, h, w, 3), dtype=torch.float32)

    def _save_synced_json(self, poses_tensor: torch.Tensor, meta: dict, bpm: float, fps: float, filename: str, debug: bool):
        """Save the synced pose data to a JSON file"""
        try:
            lib = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "..", "dance_library")
            os.makedirs(lib, exist_ok=True)
            safe = "".join(c for c in filename if c.isalnum()
                           or c in "-_").strip()
            if not safe:
                safe = "untitled_synced"
            path = os.path.join(lib, f"{safe}_synced.json")

            # Ensure tensor is on CPU and convert to list
            if isinstance(poses_tensor, torch.Tensor):
                pose_list = poses_tensor.cpu().numpy().tolist()
            else:
                pose_list = np.array(poses_tensor).tolist()

            data = {
                "title": filename,
                "metadata": {
                    "bpm": float(bpm),
                    "fps": float(fps),
                    "frame_count": len(pose_list),  # Use actual length
                    "sync_method": "auto",
                    "source_bpm": meta.get("source_bpm", 120.0),
                    "source_fps": meta.get("source_fps", 24.0)
                },
                "pose_tensor": pose_list
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            if debug:
                print(f"[BAIS1C MusicControlNet] Saved synced JSON → {path}")
        except Exception as e:
            print(
                f"[BAIS1C MusicControlNet] Failed to save JSON '{filename}': {e}")

    # --- Helper method needed for library_json path (missing in original) ---
    # Placeholder implementation, adapt path logic as needed if library loading is required.
    def _load_from_json_library(self, library_dance: str, debug: bool):
        """Placeholder for loading from JSON library."""
        print(
            f"[BAIS1C MusicControlNet] [_load_from_json_library] Library loading not implemented for '{library_dance}' in this node version.")
        return None, {}


# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_MusicControlNet": BAIS1C_MusicControlNet}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_MusicControlNet": "🎵 BAIS1C Music Control Net (Auto-Sync)"}
