import numpy as np
import librosa

class BAIS1C_SourceVideoLoader:
    """
    Source Video Loader (images+audio → sync_meta)
    - Takes images (required) and audio (optional)
    - Calculates FPS, frame count, duration, BPM (if audio given)
    - Outputs images, audio, sync_meta (DICT), ui_info (STRING)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),               # Sequence of frames (batch tensor/list)
                "audio": ("AUDIO", {"optional": True}),  # Audio is optional
            }
        }

    RETURN_TYPES  = ("IMAGE", "AUDIO", "DICT", "STRING")
    RETURN_NAMES  = ("images", "audio", "sync_meta", "ui_info")
    FUNCTION      = "load"
    CATEGORY      = "BAIS1C VACE Suite/Source"
    OUTPUT_NODE   = False

    def load(self, images, audio=None):
        # FPS estimation (if possible)
        frame_count = len(images) if hasattr(images, '__len__') else (images.shape[0] if hasattr(images, "shape") and len(images.shape) == 4 else 1)
        fps = 24.0  # Default fallback

        # Try to infer fps from audio (if waveform and sync available), else leave as default
        duration = None
        bpm = None

        # Get audio duration if possible
        if audio and isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            duration = len(waveform) / sample_rate if sample_rate > 0 else None
            # Estimate FPS if duration makes sense
            if duration and frame_count > 1:
                fps = frame_count / duration
            # BPM analysis (optional)
            bpm = self._analyze_bpm_safe(waveform, sample_rate, duration)
        else:
            waveform = None
            sample_rate = None
            duration = frame_count / fps if fps > 0 else None

        sync_meta = {
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
            "bpm": bpm,
            "sample_rate": sample_rate,
            "audio_present": bool(audio is not None),
        }
        ui_info = (
            f"{frame_count} frames @ {fps:.2f} FPS\n"
            f"Duration: {duration:.2f} s\n"
            f"BPM: {bpm if bpm is not None else 'n/a'}\n"
            f"Sample Rate: {sample_rate if sample_rate is not None else 'n/a'}\n"
        )

        return images, audio, sync_meta, ui_info

    def _analyze_bpm_safe(self, waveform, sample_rate, duration):
        try:
            if duration and duration < 3.0:
                return None
            tempo_estimates = librosa.beat.tempo(
                y=waveform, sr=sample_rate, aggregate=None, start_bpm=60, std_bpm=40
            )
            plausible_bpms = [b for b in tempo_estimates if 70 <= b <= 160]
            primary_bpm = float(min(plausible_bpms)) if plausible_bpms else (
                float(np.median(tempo_estimates)) if len(tempo_estimates) > 0 else None
            )
            while primary_bpm and primary_bpm > 150:
                primary_bpm /= 2.0
            if primary_bpm:
                return np.clip(primary_bpm, 60, 150)
            return None
        except Exception:
            return None

# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SourceVideoLoader": "🎥 BAIS1C Source Video Loader"}
