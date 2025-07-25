# BAIS1C_MusicControlNet.py
# BAIS1C VACE Suite — Music Control Net Node (Meta-Driven Refactor)
import numpy as np
import torch
import traceback

class BAIS1C_MusicControlNet:
    """
    Dance pose/music synchronizer using audio and enhanced meta analysis.
    Inputs:
      - input_pose_tensor: POSE (from PoseExtractor)
      - sync_meta: DICT (from SourceVideoLoader; must contain enhanced audio analysis)
      - audio: AUDIO (optional; used only if re-analysis is requested or missing meta)
    Outputs:
      - synced_pose_tensor: POSE (pose tensor resampled/synced to target BPM/FPS)
      - sync_report: STRING (textual summary for UI/debug)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_pose_tensor": ("POSE",),   # Primary motion tensor
                "sync_meta": ("DICT",),           # Meta dict (must include 'primary_bpm', 'total_frames', etc)
                "audio": ("AUDIO",),              # Optional, for fallback re-analysis
                "sync_mode": (["beat_aligned", "frame_aligned", "advanced"], {"default": "beat_aligned"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("POSE", "STRING")
    RETURN_NAMES = ("synced_pose_tensor", "sync_report")
    FUNCTION = "execute"
    CATEGORY = "BAIS1C VACE Suite"

    def __init__(self):
        pass

    def execute(self, input_pose_tensor, sync_meta, audio, sync_mode="beat_aligned", debug=False):
        # Defensive: all meta analysis is expected in sync_meta
        meta = dict(sync_meta) if sync_meta else {}
        
        # Convert pose tensor to numpy for processing
        if isinstance(input_pose_tensor, torch.Tensor):
            pose = input_pose_tensor.detach().cpu().numpy()
        else:
            pose = np.array(input_pose_tensor)
            
        n_frames = pose.shape[0]
        report_lines = []

        # Prefer robust fields; fallback to defaults
        primary_bpm = meta.get("primary_bpm", 120.0)
        target_fps = meta.get("fps", 24.0)
        beat_times = meta.get("beat_times", None)
        beat_strength = meta.get("beat_strength", None)
        duration = meta.get("duration", n_frames / target_fps if target_fps else n_frames / 24.0)

        report_lines.append(f"Sync Mode: {sync_mode}")
        report_lines.append(f"BPM: {primary_bpm:.2f} / FPS: {target_fps:.2f}")
        report_lines.append(f"Frames: {n_frames} / Duration: {duration:.2f}s")
        report_lines.append(f"Beat Consistency: {meta.get('beat_consistency', 'n/a')}")
        report_lines.append(f"Swing: {meta.get('rhythm_patterns', {}).get('has_swing', False)}")

        # Choose sync strategy
        if sync_mode == "beat_aligned" and beat_times is not None and len(beat_times) > 1:
            synced_pose = self._sync_to_beat(pose, beat_times, duration, target_fps, debug)
            report_lines.append("Beat-aligned sync applied.")
        elif sync_mode == "frame_aligned":
            synced_pose = self._sync_to_frames(pose, int(duration * target_fps))
            report_lines.append("Frame-aligned (direct resample) sync applied.")
        elif sync_mode == "advanced":
            synced_pose = self._advanced_sync(pose, meta, debug)
            report_lines.append("Advanced sync with multiple factors applied.")
        else:
            synced_pose = pose
            report_lines.append("No resampling: pose tensor passed through unchanged.")

        # Add music reactivity if additional fields are present
        if "freq_bands" in meta and "beat_strength" in meta:
            try:
                synced_pose = self._apply_music_reactivity(
                    synced_pose,
                    meta,
                    debug=debug
                )
                report_lines.append("Music reactivity enhancements applied.")
            except Exception as e:
                report_lines.append(f"[WARN] Music reactivity failed: {e}")
                if debug:
                    traceback.print_exc()

        # Convert back to torch tensor
        if isinstance(input_pose_tensor, torch.Tensor):
            synced_pose_tensor = torch.from_numpy(synced_pose).float()
        else:
            synced_pose_tensor = synced_pose

        report_lines.append("Sync complete.")
        return (synced_pose_tensor, "\n".join(report_lines))

    def _sync_to_beat(self, pose, beat_times, duration, fps, debug=False):
        """Resample pose tensor so each beat aligns to a keyframe."""
        n_beats = len(beat_times)
        total_frames = int(duration * fps)
        
        if total_frames < 1:
            total_frames = 1
            
        beat_indices = np.linspace(0, total_frames - 1, n_beats).astype(int)
        orig_indices = np.linspace(0, pose.shape[0] - 1, n_beats).astype(int)
        synced_pose = np.zeros((total_frames, *pose.shape[1:]), dtype=pose.dtype)
        
        # Linear interpolation between keyframes at beats
        for i in range(n_beats - 1):
            start_idx = beat_indices[i]
            end_idx = beat_indices[i + 1]
            p0 = pose[orig_indices[i]]
            p1 = pose[orig_indices[i + 1]]
            
            for f in range(start_idx, end_idx):
                t = (f - start_idx) / max(1, end_idx - start_idx)
                synced_pose[f] = (1 - t) * p0 + t * p1
                
        # Fill after last beat
        if n_beats >= 2:
            synced_pose[beat_indices[-1]:] = pose[orig_indices[-1]]
        else:
            synced_pose[:] = pose[0] if len(pose) > 0 else np.zeros_like(synced_pose[0])
            
        if debug:
            print(f"[MusicControlNet] Beat-aligned sync: {pose.shape[0]} → {total_frames} frames")
            
        return synced_pose

    def _sync_to_frames(self, pose, target_frames):
        """Resample pose tensor to match target frame count."""
        if target_frames < 1:
            target_frames = 1
            
        orig_frames = pose.shape[0]
        if orig_frames == target_frames:
            return pose
            
        indices = np.linspace(0, orig_frames - 1, target_frames)
        indices = np.clip(indices, 0, orig_frames - 1).astype(int)
        return pose[indices]

    def _advanced_sync(self, pose, meta, debug=False):
        """Advanced sync using multiple audio features"""
        # Get various sync parameters
        beat_times = meta.get("beat_times", [])
        onset_times = meta.get("onsets", {}).get("combined_times", [])
        duration = meta.get("duration", 1.0)
        fps = meta.get("fps", 24.0)
        
        total_frames = int(duration * fps)
        if total_frames < 1:
            total_frames = 1
            
        synced_pose = np.zeros((total_frames, *pose.shape[1:]), dtype=pose.dtype)
        
        # Combine beat and onset information for keyframe placement
        all_events = list(beat_times) + list(onset_times)
        all_events = sorted(set(all_events))  # Remove duplicates and sort
        
        if len(all_events) > 1:
            # Map events to frame indices
            event_frames = [int((event / duration) * total_frames) for event in all_events]
            event_frames = [max(0, min(total_frames - 1, f)) for f in event_frames]
            
            # Map to original pose indices
            orig_indices = np.linspace(0, pose.shape[0] - 1, len(event_frames)).astype(int)
            
            # Interpolate between event keyframes
            for i, frame_idx in enumerate(event_frames):
                if i < len(orig_indices):
                    synced_pose[frame_idx] = pose[orig_indices[i]]
            
            # Fill gaps with interpolation
            for i in range(len(event_frames) - 1):
                start_frame = event_frames[i]
                end_frame = event_frames[i + 1]
                start_pose = synced_pose[start_frame]
                end_pose = synced_pose[end_frame]
                
                for f in range(start_frame + 1, end_frame):
                    t = (f - start_frame) / (end_frame - start_frame)
                    synced_pose[f] = (1 - t) * start_pose + t * end_pose
        else:
            # Fallback to simple resampling
            synced_pose = self._sync_to_frames(pose, total_frames)
            
        if debug:
            print(f"[MusicControlNet] Advanced sync: {len(all_events)} events, {total_frames} frames")
            
        return synced_pose

    def _apply_music_reactivity(self, pose, meta, debug=False):
        """Apply music reactivity based on frequency bands and beat strength"""
        freq_bands = meta.get("freq_bands", {})
        beat_strength = meta.get("beat_strength", [])
        n_frames = pose.shape[0]
        
        # Get frequency band energies
        bass = freq_bands.get("bass", {}).get("energy", np.zeros(n_frames))
        mid = freq_bands.get("mid", {}).get("energy", np.zeros(n_frames))
        highs = freq_bands.get("highs", {}).get("energy", np.zeros(n_frames))
        
        # Ensure arrays match frame count
        bass = self._ensure_array_length(bass, n_frames)
        mid = self._ensure_array_length(mid, n_frames)
        highs = self._ensure_array_length(highs, n_frames)
        beat_strength = self._ensure_array_length(beat_strength, n_frames)
        
        # Create modulated pose copy
        pose_mod = pose.copy()
        
        # Apply reactivity frame by frame
        for f in range(n_frames):
            # Calculate modulation strength
            bass_strength = bass[f] * 0.03
            mid_strength = mid[f] * 0.02
            high_strength = highs[f] * 0.015
            beat_boost = beat_strength[f] * 0.025
            
            # Apply to different body parts
            if pose.shape[1] > 12:  # Ensure we have enough keypoints
                # Hip bounce (bass response)
                if pose.shape[1] > 12:
                    pose_mod[f, 11, 1] += bass_strength + beat_boost * 0.8  # Left hip Y
                    pose_mod[f, 12, 1] += bass_strength + beat_boost * 0.8  # Right hip Y
                
                # Shoulder movement (mid response)
                if pose.shape[1] > 6:
                    pose_mod[f, 5, 1] += mid_strength * np.sin(f * 0.1)  # Left shoulder
                    pose_mod[f, 6, 1] += mid_strength * np.sin(f * 0.1 + np.pi)  # Right shoulder
                
                # Hand/wrist jitter (high frequency response)
                if pose.shape[1] > 10:
                    noise_x = np.random.uniform(-1, 1) * high_strength * 0.5
                    noise_y = np.random.uniform(-1, 1) * high_strength * 0.5
                    pose_mod[f, 9, 0] += noise_x   # Left wrist X
                    pose_mod[f, 10, 0] += noise_x  # Right wrist X
                    pose_mod[f, 9, 1] += noise_y   # Left wrist Y
                    pose_mod[f, 10, 1] += noise_y  # Right wrist Y
                
                # Head nod (beat response)
                if pose.shape[1] > 0:
                    pose_mod[f, 0, 1] += beat_boost * np.sin(f * 0.2)  # Nose/head Y
        
        # Keep poses in valid range (0-1 for normalized coordinates)
        pose_mod = np.clip(pose_mod, 0.0, 1.0)
        
        if debug:
            print(f"[MusicControlNet] Music reactivity applied to {n_frames} frames")
            print(f"  Bass RMS: {np.mean(bass):.3f}, Mid RMS: {np.mean(mid):.3f}, Highs RMS: {np.mean(highs):.3f}")
        
        return pose_mod

    def _ensure_array_length(self, arr, target_length):
        """Ensure array matches target length"""
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        
        if len(arr) == 0:
            return np.zeros(target_length)
        
        if len(arr) == target_length:
            return arr
        
        # Resample to target length
        old_indices = np.linspace(0, len(arr) - 1, len(arr))
        new_indices = np.linspace(0, len(arr) - 1, target_length)
        return np.interp(new_indices, old_indices, arr)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "BAIS1C_MusicControlNet": BAIS1C_MusicControlNet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_MusicControlNet": "BAIS1C Music ControlNet (Meta Sync)",
}

# Self-test function
def _test_music_control_net():
    """Test MusicControlNet functionality"""
    print("🔍 Testing BAIS1C_MusicControlNet...")
    
    # Test data
    pose = np.random.rand(240, 128, 2)  # 10s @ 24fps
    sync_meta = {
        "primary_bpm": 120.0,
        "fps": 24.0,
        "duration": 10.0,
        "beat_times": np.linspace(0, 10, 21).tolist(),
        "beat_consistency": 0.95,
        "beat_strength": np.random.rand(240).tolist(),
        "rhythm_patterns": {"has_swing": False},
        "freq_bands": {
            "bass": {"energy": np.random.rand(240).tolist()},
            "mid": {"energy": np.random.rand(240).tolist()},
            "highs": {"energy": np.random.rand(240).tolist()},
        }
    }
    
    node = BAIS1C_MusicControlNet()
    
    try:
        # Test 1: Beat-aligned sync
        synced, report = node.execute(pose, sync_meta, None, sync_mode="beat_aligned", debug=True)
        
        assert isinstance(synced, np.ndarray), "Output should be numpy array"
        assert synced.shape[1:] == (128, 2), f"Wrong pose shape: {synced.shape}"
        assert isinstance(report, str), "Report should be string"
        
        print("✅ Beat-aligned sync test passed")
        print(f"   Input: {pose.shape} → Output: {synced.shape}")
        
        # Test 2: Frame-aligned sync
        synced2, report2 = node.execute(pose, sync_meta, None, sync_mode="frame_aligned", debug=True)
        
        assert isinstance(synced2, np.ndarray), "Output should be numpy array"
        print("✅ Frame-aligned sync test passed")
        
        # Test 3: Advanced sync
        synced3, report3 = node.execute(pose, sync_meta, None, sync_mode="advanced", debug=True)
        
        assert isinstance(synced3, np.ndarray), "Output should be numpy array"
        print("✅ Advanced sync test passed")
        
        print("✅ All MusicControlNet tests passed!")
        print(f"   Report length: {len(report)} chars")
        return True
        
    except Exception as e:
        print(f"❌ MusicControlNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Inline test function (for standalone/CLI dev testing)
if __name__ == "__main__":
    _test_music_control_net()