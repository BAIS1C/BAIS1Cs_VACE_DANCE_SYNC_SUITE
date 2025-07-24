# BAIS1C_SourceVideoLoader.py
# Minimal, Composable Video Loader Node for VACE Dance Sync Suite
import numpy as np
import traceback

# Import your enhanced audio analyzer here
from enhanced_audio_analysis import EnhancedAudioAnalyzer  # Update import path as needed

class BAIS1C_SourceVideoLoader:
    """
    Minimal pass-through loader for images/audio + robust audio analysis.
    Input: images, audio (optional), video_info (dict/meta)
    Output: images, audio, sync_meta (dict with audio/BPM etc), ui_info (summary)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),          # Frame sequence from VHS_LoadVideo or similar
                "audio": ("AUDIO",),           # Optional, but required for BPM/etc
                "video_info": ("DICT",)        # Metadata dict from upstream node
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
        # Defensive copy to avoid mutating input dict
        sync_meta = dict(video_info) if video_info else {}
        target_fps = sync_meta.get("fps", 24.0)
        audio_analysis = None
        # Only analyze if audio is present
        if audio and audio.get("waveform") is not None and audio.get("sample_rate") is not None:
            audio_analysis = self.analyze_audio(audio, target_fps)
            if audio_analysis:
                sync_meta.update(audio_analysis)
                sync_meta["audio_analysis_performed"] = True
            else:
                sync_meta["audio_analysis_performed"] = False
        else:
            sync_meta["audio_analysis_performed"] = False

        # Clean UI string for panel/debug
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


# ComfyUI Node registration block (add to your nodes/__init__.py if needed)
NODE_CLASS_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": "BAIS1C Source Video Loader (BPM/Meta)",
}


# Inline test function (sanity check)
if __name__ == "__main__":
    # Fake test data (minimal test, expand as needed)
    sr = 44100
    test_waveform = np.sin(2 * np.pi * 2 * np.linspace(0, 1, sr))
    test_audio = {"waveform": test_waveform, "sample_rate": sr}
    test_images = "IMAGE_PLACEHOLDER"
    test_meta = {"fps": 24.0, "duration": 1.0, "original_frame_count": 24}
    node = BAIS1C_SourceVideoLoader()
    out = node.execute(test_images, test_audio, test_meta)
    print("[TEST OUTPUT]", out[2])  # Print sync_meta

