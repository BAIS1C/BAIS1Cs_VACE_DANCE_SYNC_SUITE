# BAIS1C_SourceVideoLoader.py
# Minimal, Composable Video Loader Node for VACE Dance Sync Suite
import numpy as np
import traceback

from .enhanced_audio_analysis import EnhancedAudioAnalyzer

def ensure_dict(input_obj):
    """Utility to robustly convert VHS/VideoHelperSuite video_info to dict."""
    if input_obj is None:
        return {}
    if isinstance(input_obj, dict):
        return dict(input_obj)
    # Handle classes with .to_dict() or .as_dict()
    if hasattr(input_obj, "to_dict"):
        return dict(input_obj.to_dict())
    if hasattr(input_obj, "as_dict"):
        return dict(input_obj.as_dict())
    # Try vars()
    if hasattr(input_obj, "__dict__"):
        return dict(vars(input_obj))
    # Try items()
    if hasattr(input_obj, "items"):
        return dict(input_obj.items())
    # Last resort: string conversion
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
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),          # Frame sequence from VHS_LoadVideo or similar
                "audio": ("AUDIO",),           # Optional, but required for BPM/etc
                "video_info": ("VHS_VIDEOINFO",),       # Metadata dict (now robust to VHS_VIDEOINFO etc)
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "DICT", "STRING")
    RETURN_NAMES = ("images", "audio", "sync_meta", "ui_info")
    FUNCTION = "execute"
    CATEGORY = "BAIS1C VACE Suite"

    def __init__(self):
        pass

    def analyze_audio(self, audio, target_fps):
        if not audio or "waveform" not in audio or "sample_rate" not in audio:
            return None
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        try:
            analyzer = EnhancedAudioAnalyzer(sample_rate)
            return analyzer.analyze_comprehensive(waveform, target_fps=target_fps, debug=False)
        except Exception as e:
            print(f"[BAIS1C_SourceVideoLoader] Audio analysis failed: {e}")
            traceback.print_exc()
            return None

    def execute(self, images, audio, video_info):
        # Fix: Accept VHS_VIDEOINFO and dict-like input robustly
        sync_meta = ensure_dict(video_info)
        target_fps = float(sync_meta.get("fps", 24.0))
        audio_analysis = None

        if audio and audio.get("waveform") is not None and audio.get("sample_rate") is not None:
            audio_analysis = self.analyze_audio(audio, target_fps)
            if audio_analysis:
                sync_meta.update(audio_analysis)
                sync_meta["audio_analysis_performed"] = True
            else:
                sync_meta["audio_analysis_performed"] = False
        else:
            sync_meta["audio_analysis_performed"] = False

        # UI info for ComfyUI summary panel/debug
        lines = []
        if audio_analysis:
            lines.append(f"BPM: {audio_analysis.get('primary_bpm', 0):.2f} (Conf: {audio_analysis.get('bpm_confidence', 0):.2f})")
            lines.append(f"Duration: {audio_analysis.get('duration', 0):.2f}s, Frames: {audio_analysis.get('total_frames', 0)}")
            lines.append(f"Beat Consistency: {audio_analysis.get('beat_consistency', 0):.2f}")
            if "rhythm_patterns" in audio_analysis:
                swing = audio_analysis["rhythm_patterns"].get("has_swing", False)
                lines.append("Rhythm: Swing" if swing else "Rhythm: Straight")
            if "freq_bands" in audio_analysis:
                bands = audio_analysis["freq_bands"]
                for b in ["bass", "mid", "highs"]:
                    if b in bands:
                        rms = bands[b].get("rms", 0)
                        lines.append(f"{b.capitalize()} RMS: {rms:.3f}")
        else:
            lines.append("No audio analysis performed.")
        ui_info = "\n".join(lines)

        return (images, audio, sync_meta, ui_info)

# ComfyUI Node registration block
NODE_CLASS_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": "BAIS1C Source Video Loader (BPM/Meta)",
}

# Inline test function (for standalone/CLI dev testing)
if __name__ == "__main__":
    # Test with minimal dummy data
    sr = 44100
    test_waveform = np.sin(2 * np.pi * 2 * np.linspace(0, 1, sr))
    test_audio = {"waveform": test_waveform, "sample_rate": sr}
    test_images = "IMAGE_PLACEHOLDER"
    # Simulate dict-like and object input for video_info
    test_meta_dict = {"fps": 24.0, "duration": 1.0, "original_frame_count": 24}
    class DummyInfo:  # Mimics VHS_VIDEOINFO
        def __init__(self):
            self.fps = 24.0
            self.duration = 1.0
            self.original_frame_count = 24
        def to_dict(self):
            return {"fps": self.fps, "duration": self.duration, "original_frame_count": self.original_frame_count}
    test_meta_obj = DummyInfo()
    node = BAIS1C_SourceVideoLoader()
    print("[TEST OUTPUT - dict]", node.execute(test_images, test_audio, test_meta_dict)[2])  # sync_meta
    print("[TEST OUTPUT - obj]", node.execute(test_images, test_audio, test_meta_obj)[2])  # sync_meta

