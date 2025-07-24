from pathlib import Path
import os
import torch
import subprocess
import decord
import librosa
import numpy as np
import soundfile as sf
import tempfile
import shutil

class BAIS1C_SourceVideoLoader:
    """
    BAIS1C VACE Dance Sync Suite - Source Video Loader

    Loads video, extracts and displays sync-relevant metadata (with video preview).
    Outputs: video_obj, audio_obj (waveform), and sync_meta (dict).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),  # ComfyUI Video object, path, or upload
            }
        }

    RETURN_TYPES = ("VIDEO", "AUDIO", "DICT", "STRING")
    RETURN_NAMES = ("video_obj", "audio_obj", "sync_meta","ui_info")
    OUTPUT_NODE = True
    FUNCTION = "load"
    CATEGORY = "BAIS1C VACE Suite/Source"

    def load(self, video):
        """
        Load video, extract audio, analyze for BPM and metadata, output all.
        """
        # ------------------------------------------------------------------
        # 1. Resolve video path
        # ------------------------------------------------------------------
        if hasattr(video, "video_path"):
            video_path = video.video_path
        elif hasattr(video, "path"):           # NEW: VideoFromFile wrapper
            video_path = video.path
        elif isinstance(video, str):
            video_path = video
        else:
            raise ValueError(f"Unsupported video input type: {type(video)}.")

        # Video properties
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = float(vr.get_avg_fps())
        frame_count = len(vr)
        duration = frame_count / fps if fps > 0 else 0.0

        # Aspect ratio check
        height, width = vr[0].shape[:2]
        aspect_ratio = width / height if height != 0 else 0
        is_16_9 = abs(aspect_ratio - 16/9) < 0.05
        is_9_16 = abs(aspect_ratio - 9/16) < 0.05

        # Audio
        audio_obj = self._extract_audio(video_path)
        bpm = self._analyze_bpm_enhanced(audio_obj["waveform"], audio_obj["sample_rate"], duration)

        # Build sync_meta (summary for pose/music nodes)
        sync_meta = {
            "file_name": os.path.basename(video_path),
            "video_path": video_path,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "aspect_status": (
                "16:9" if is_16_9 else
                "9:16" if is_9_16 else
                "Non-standard"
            ),
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "bpm": bpm,
        }

        # Compose info string for UI display
        info_lines = [
            f"File: {sync_meta['file_name']}",
            f"Resolution: {width}x{height} ({aspect_ratio:.2f})",
            f"Aspect: {sync_meta['aspect_status']} " +
              ("✅" if sync_meta["aspect_status"] in ("16:9","9:16") else "⚠️ WARNING: Non-optimal, may cause warping/stretching"),
            f"FPS: {fps:.2f}",
            f"Duration: {duration:.2f}s, Frames: {frame_count}",
            f"BPM: {bpm:.2f}",
        ]
        sync_meta["info_string"] = "\n".join(info_lines)

        # For future: optionally push info_string to a ComfyUI text display node.

        print("[BAIS1C VACE Suite] " + sync_meta["info_string"].replace('\n', ' | '))

        return (video, audio_obj, sync_meta, sync_meta["info_string"])

    def _extract_audio(self, video_path):
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found in PATH.")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', tmp_path
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
            waveform, sample_rate = sf.read(tmp_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform)) * 0.95
            return {"waveform": waveform, "sample_rate": sample_rate}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _check_ffmpeg(self):
        return shutil.which('ffmpeg') is not None

    def _analyze_bpm_enhanced(self, waveform, sample_rate, duration):
        try:
            tempo_estimates = librosa.beat.tempo(
                y=waveform, sr=sample_rate, aggregate=None,
                start_bpm=60, std_bpm=40
            )
            bpm = float(tempo_estimates[0]) if len(tempo_estimates) > 0 else 120.0
            return bpm
        except Exception:
            return 120.0

NODE_CLASS_MAPPINGS = {"BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SourceVideoLoader": "🎥 BAIS1C Source Video Loader"}
