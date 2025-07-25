# BAIS1C_SourceVideoLoader.py
# Minimal, Composable Video Loader Node for DWPose-23 Dance Sync Suite

import numpy as np
import traceback

from .enhanced_audio_analysis import EnhancedAudioAnalyzer

def ensure_dict(input_obj):
    """Utility to robustly convert VHS/VideoHelperSuite video_info to dict."""
    if input_obj is None:
        return {}
    if isinstance(input_obj, dict):
        return dict(input_obj)
    if hasattr(input_obj, "to_dict"):
        return dict(input_obj.to_dict())
    if hasattr(input_obj, "as_dict"):
        return dict(input_obj.as_dict())
    if hasattr(input_obj, "__dict__"):
        obj_dict = vars(input_obj)
        filtered_dict = {k: v for k, v in obj_dict.items() if not k.startswith('_') and not callable(v)}
        return filtered_dict
    if hasattr(input_obj, "items"):
        return dict(input_obj.items())
    try:
        return dict(input_obj)
    except Exception:
        pass
    return {}

class BAIS1C_SourceVideoLoader:
    """
    Minimal pass-through loader for images/audio + robust audio analysis.
    Input: images, audio (optional), video_info (dict/meta from VHS/VideoHelper/etc)
    Output: images, audio, sync_meta (dict with audio/BPM etc), ui_info (summary)
    Outputs are tailored for DWPose UCoco23 (23-point) pipeline.
    If BPM cannot be robustly detected, the meta will use:
      "primary_bpm": None
      "bpm_confidence": 0.0
      "bpm_status": "unavailable"
    Downstream sync nodes should always check for None or bpm_confidence < threshold.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),          # Frame sequence from VHS_LoadVideo or similar
                "audio": ("AUDIO",),           # Optional, but required for BPM/etc
                "video_info": ("VHS_VIDEOINFO",), # Metadata dict/object (robust to class or dict)
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "DICT", "STRING")
    RETURN_NAMES = ("images", "audio", "sync_meta", "ui_info")
    FUNCTION = "execute"
    CATEGORY = "BAIS1C VACE Suite"

    def __init__(self):
        pass

    def analyze_audio(self, audio, target_fps):
        """Comprehensive audio analysis — robust BPM detection for sync."""
        if not audio or "waveform" not in audio or "sample_rate" not in audio:
            return None
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if hasattr(waveform, 'detach'):
            waveform = waveform.detach().cpu().numpy()
        try:
            analyzer = EnhancedAudioAnalyzer(sample_rate)
            analysis = analyzer.analyze_comprehensive(waveform, target_fps=target_fps, debug=False)
            expected_frames = analysis.get('total_frames', 0)
            # Defensive: check for real BPM
            bpm_val = analysis.get('primary_bpm', None)
            bpm_conf = analysis.get('bpm_confidence', 0.0)
            if bpm_val is None or bpm_conf < 0.01:
                # Override fields for unknown BPM
                analysis['primary_bpm'] = None
                analysis['bpm_confidence'] = 0.0
                analysis['bpm_status'] = "unavailable"
            else:
                analysis['bpm_status'] = "detected"
            for key in ['beat_strength', 'beat', 'bass', 'energy']:
                if key in analysis:
                    actual_length = len(analysis[key])
                    if actual_length != expected_frames:
                        print(f"[SourceVideoLoader] WARNING: {key} array size mismatch: {actual_length} != {expected_frames}")
            return analysis
        except Exception as e:
            print(f"[BAIS1C_SourceVideoLoader] Audio analysis failed: {e}")
            traceback.print_exc()
            return None

    def execute(self, images, audio, video_info):
        """Main execution with consistent sync_meta handling (23-point DWPose)"""
        sync_meta = ensure_dict(video_info)
        # Extract or default key parameters
        target_fps = float(sync_meta.get("loaded_fps", sync_meta.get("source_fps", sync_meta.get("fps", 24.0))))
        duration = float(sync_meta.get("loaded_duration", sync_meta.get("source_duration", sync_meta.get("duration", 1.0))))
        frame_count = int(sync_meta.get("loaded_frame_count", sync_meta.get("source_frame_count", sync_meta.get("frame_count", 24))))
        # Standardized meta
        sync_meta.update({
            "fps": target_fps,
            "duration": duration,
            "frame_count": frame_count,
            "audio_present": audio is not None and "waveform" in audio,
            "skeleton_layout": "UCoco23",
            "points": 23,
        })
        # Perform audio analysis if available
        audio_analysis = None
        if audio and audio.get("waveform") is not None and audio.get("sample_rate") is not None:
            audio_analysis = self.analyze_audio(audio, target_fps)
            if audio_analysis:
                sync_meta.update(audio_analysis)
                sync_meta["audio_analysis_performed"] = True
                # Diagnostics for BPM
                bpm_val = sync_meta.get("primary_bpm", None)
                bpm_conf = sync_meta.get("bpm_confidence", 0.0)
                if bpm_val is None or bpm_conf < 0.01:
                    print(f"[SourceVideoLoader] WARNING: BPM unavailable or confidence too low (value={bpm_val}, conf={bpm_conf:.2f}).")
                required_fields = ['primary_bpm', 'total_frames', 'beat_strength', 'freq_bands']
                missing_fields = [field for field in required_fields if field not in sync_meta]
                if missing_fields:
                    print(f"[SourceVideoLoader] WARNING: Missing audio analysis fields: {missing_fields}")
            else:
                sync_meta["audio_analysis_performed"] = False
                print("[SourceVideoLoader] Audio analysis failed — using default sync values.")
                sync_meta.update(self._default_audio_fields(frame_count))
        else:
            sync_meta["audio_analysis_performed"] = False
            print("[SourceVideoLoader] No audio provided — using default sync values.")
            sync_meta.update(self._default_audio_fields(frame_count))

        # UI info summary
        ui_info = self._create_ui_info(sync_meta, audio_analysis)
        return (images, audio, sync_meta, ui_info)

    def _default_audio_fields(self, frame_count):
        """Return robust default sync meta for missing/failed audio."""
        return {
            "primary_bpm": None,                # Not a fake number!
            "bpm_confidence": 0.0,
            "bpm_status": "unavailable",        # Explicit "not available"
            "beat_strength": np.zeros(frame_count).tolist(),
            "beat_times": [],
            "freq_bands": {
                "bass": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0},
                "mid": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0},
                "highs": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0}
            },
        }

    def _create_ui_info(self, sync_meta, audio_analysis):
        """Create summary string for UI. Clarifies 23-point DWPose output."""
        lines = []
        lines.append(f"📺 Video: {sync_meta.get('frame_count', 0)} frames @ {sync_meta.get('fps', 0):.1f} FPS")
        lines.append(f"⏱️  Duration: {sync_meta.get('duration', 0):.2f}s")
        lines.append(f"🦴 Skeleton: UCoco23 (23-point, DWPose only)")
        bpm_val = sync_meta.get("primary_bpm", None)
        bpm_conf = sync_meta.get("bpm_confidence", 0.0)
        if bpm_val is not None and bpm_conf > 0.01:
            lines.append(f"🎵 BPM: {bpm_val:.2f} (Conf: {bpm_conf:.2f})")
            lines.append(f"🎼 Total Frames: {audio_analysis.get('total_frames', 0) if audio_analysis else 0}")
            rhythm = audio_analysis.get("rhythm_patterns", {}) if audio_analysis else {}
            if rhythm:
                swing_status = "Swing" if rhythm.get("has_swing", False) else "Straight"
                lines.append(f"🎶 Rhythm: {swing_status} (Ratio: {rhythm.get('swing_ratio', 1.0):.2f})")
                lines.append(f"🎯 Groove Strength: {rhythm.get('groove_strength', 0):.2f}")
            beat_count = len(audio_analysis.get('beat_times', [])) if audio_analysis else 0
            lines.append(f"🥁 Beats Detected: {beat_count}")
        else:
            lines.append("🔇 BPM unavailable or confidence too low. No beat-based sync possible.")
        return "\n".join(lines)

# Node registration
NODE_CLASS_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": "BAIS1C Source Video Loader (BPM/Meta, DWPose23)",
}

# Self-test
def _test_source_video_loader():
    print("[TEST] BAIS1C_SourceVideoLoader")
    sr = 44100
    duration = 2.0
    test_waveform = np.sin(2 * np.pi * 2 * np.linspace(0, duration, int(sr * duration)))
    test_audio = {"waveform": test_waveform, "sample_rate": sr}
    test_images = np.random.randint(0, 255, (48, 480, 640, 3), dtype=np.uint8)  # 2s @ 24fps
    test_meta_dict = {"fps": 24.0, "duration": duration, "frame_count": 48}
    node = BAIS1C_SourceVideoLoader()
    images, audio, sync_meta, ui_info = node.execute(test_images, test_audio, test_meta_dict)
    print(ui_info)
    assert isinstance(sync_meta, dict)
    assert "primary_bpm" in sync_meta
    assert sync_meta["points"] == 23
    print("[TEST PASSED] SourceVideoLoader")

# Uncomment to self-test
# _test_source_video_loader()
