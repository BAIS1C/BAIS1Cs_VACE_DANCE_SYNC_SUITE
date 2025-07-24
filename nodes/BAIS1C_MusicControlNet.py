# BAIS1C_MusicControlNet.py
# BAIS1C VACE Suite — Music Control Net Node (Meta-Driven Refactor)
import numpy as np
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
                "sync_mode": (["beat_aligned", "frame_aligned", "advanced"],),  # Optional mode selector
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
        else:
            synced_pose = pose
            report_lines.append("No resampling: pose tensor passed through unchanged.")

        # (OPTIONAL) Add music reactivity if additional fields are present
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

        report_lines.append("Sync complete.")
        return (synced_pose, "\n".join(report_lines))

    def _sync_to_beat(self, pose, beat_times, duration, fps, debug=False):
        """Resample pose tensor so each beat aligns to a keyframe."""
        n_beats = len(beat_times)
        total_frames = int(duration * fps)
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
            synced_pose[:] = pose[0]
        if debug:
            print("[MusicControlNet] Beat-aligned sync complete.")
        return synced_pose

    def _sync_to_frames(self, pose, target_frames):
        """Resample pose tensor to match target frame count."""
        orig_frames = pose.shape[0]
        idxs = np.linspace(0, orig_frames - 1, target_frames).astype(int)
        return pose[idxs]

    def _apply_music_reactivity(self, pose, meta, debug=False):
        """Example: Modulate pose based on frequency bands (simple demo logic)."""
        # This is a placeholder. For real reactivity, port your EnhancedAudioAnalyzer's logic here!
        freq_bands = meta.get("freq_bands", {})
        n_frames = pose.shape[0]
        bass = freq_bands.get("bass", {}).get("energy", np.zeros(n_frames))
        mid = freq_bands.get("mid", {}).get("energy", np.zeros(n_frames))
        highs = freq_bands.get("highs", {}).get("energy", np.zeros(n_frames))
        # Example: Add subtle bounce to hips (y) and wrists (x) based on bass/mid
        pose_mod = pose.copy()
        for f in range(n_frames):
            strength = bass[f] * 0.03 + mid[f] * 0.02 + highs[f] * 0.01
            # Hip bounce
            if pose.shape[1] > 12:
                pose_mod[f, 11, 1] += strength * 0.8  # Left hip (y)
                pose_mod[f, 12, 1] += strength * 0.8  # Right hip (y)
            # Wrist jitter (x)
            if pose.shape[1] > 10:
                pose_mod[f, 9, 0] += highs[f] * 0.01 * np.random.uniform(-1, 1)
                pose_mod[f, 10, 0] += highs[f] * 0.01 * np.random.uniform(-1, 1)
        if debug:
            print("[MusicControlNet] Music reactivity applied (bass/mid/highs).")
        return pose_mod


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "BAIS1C_MusicControlNet": BAIS1C_MusicControlNet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_MusicControlNet": "BAIS1C Music ControlNet (Meta Sync)",
}

# Inline test/demo (run as python BAIS1C_MusicControlNet.py for self-check)
if __name__ == "__main__":
    # Fake test data (minimal check)
    pose = np.zeros((240, 23, 2))  # 10s @ 24fps
    sync_meta = {
        "primary_bpm": 120.0,
        "fps": 24.0,
        "duration": 10.0,
        "beat_times": np.linspace(0, 10, 21).tolist(),
        "beat_consistency": 0.95,
        "rhythm_patterns": {"has_swing": False},
        "freq_bands": {
            "bass": {"energy": np.random.rand(240)},
            "mid": {"energy": np.random.rand(240)},
            "highs": {"energy": np.random.rand(240)},
        }
    }
    node = BAIS1C_MusicControlNet()
    synced, report = node.execute(pose, sync_meta, None, sync_mode="beat_aligned", debug=True)
    print("[TEST REPORT]\n", report)

