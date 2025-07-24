import os
import decord
import librosa
import numpy as np
import soundfile as sf
import tempfile
import shutil

class BAIS1C_SourceVideoLoader:
    """
    BAIS1C VACE Dance Sync Suite - Source Video Loader (UI Version)
    Loads a video (upload, drag/drop, or file path), extracts metadata and audio,
    and displays info and preview in the ComfyUI node panel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",)
            }
        }

    RETURN_TYPES = ("VIDEO", "DICT", "DICT")
    RETURN_NAMES = (
        "video_obj",    # Pass to pose extractor or VACE
        "audio_obj",    # {"waveform":..., "sample_rate":...}
        "sync_meta",    # Dict: metadata for downstream nodes/UI
    )
    FUNCTION = "load"
    CATEGORY = "BAIS1C VACE Suite/Source"

    def load(self, video):
        if hasattr(video, "video_path"):
            video_path = video.video_path
        elif isinstance(video, str):
            video_path = video
        else:
            raise ValueError(f"Unsupported video input type: {type(video)}.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Extract video properties
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = float(vr.get_avg_fps())
        frame_count = len(vr)
        duration = frame_count / fps if fps > 0 else 0
        height, width = vr[0].shape[:2]
        aspect_ratio = width / height
        res_str = f"{width}x{height}"

        # Aspect check for UI
        is_16_9 = abs(aspect_ratio - 16/9) < 0.1
        is_9_16 = abs(aspect_ratio - 9/16) < 0.1

        aspect_status = "OK"
        aspect_color = "lime"
        aspect_note = ""
        if is_16_9:
            aspect_note = "16:9 (Optimal for landscape)"
        elif is_9_16:
            aspect_note = "9:16 (Optimal for portrait)"
        else:
            aspect_status = "WARNING"
            aspect_color = "orange"
            aspect_note = "Non-standard! Stretch/warp possible."

        # Audio extraction (robust)
        audio_obj = self._extract_audio(video_path)

        # BPM analysis (fast/robust)
        bpm = self._analyze_bpm(audio_obj["waveform"], audio_obj["sample_rate"], duration)

        sync_meta = {
            "file_name": os.path.basename(video_path),
            "resolution": res_str,
            "aspect_ratio": f"{aspect_ratio:.2f}",
            "aspect_status": aspect_status,
            "aspect_note": aspect_note,
            "aspect_color": aspect_color,
            "fps": fps,
            "duration": duration,
            "frame_count": frame_count,
            "bpm": bpm,
        }

        return (video, audio_obj, sync_meta)

    # --- UI/Display methods (for ComfyUI node panel) ---

    @classmethod
    def IS_PREVIEW(cls):
        return True

    def get_extra_ui(self, inputs, outputs):
        # Try to fetch sync_meta dict from outputs for panel display
        meta = outputs.get("sync_meta", None)
        if not meta:
            return "<i>No video loaded or no metadata found.</i>"
        color = meta.get("aspect_color", "lime")
        return f"""
        <div style="font-size:15px;line-height:1.7;">
            <b>File:</b> {meta['file_name']}<br>
            <b>Resolution:</b> {meta['resolution']}<br>
            <b>Aspect Ratio:</b>
                <span style="color:{color};font-weight:bold">{meta['aspect_ratio']}</span>
                <span style="color:gray;font-size:12px;">({meta['aspect_note']})</span><br>
            <b>FPS:</b> {meta['fps']:.2f} &nbsp;
            <b>Duration:</b> {meta['duration']:.2f}s &nbsp;
            <b>Frames:</b> {meta['frame_count']}<br>
            <b>BPM:</b> {meta['bpm']:.1f}<br>
            <b>Sync Status:</b> <span style="color:{color};font-weight:bold">{meta['aspect_status']}</span>
            {"<span style='color:orange'>&#9888; Non-standard aspect! Use 16:9 or 9:16 for best results.</span>" if meta['aspect_status']=="WARNING" else ""}
        </div>
        """

    def get_preview(self, inputs, outputs):
        # This tells ComfyUI which output is the previewable video.
        return outputs.get("video_obj", None)

    # --- Helpers ---

    def _extract_audio(self, video_path):
        if not shutil.which('ffmpeg'):
            raise RuntimeError("FFmpeg not found in PATH.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '1', tmp_path
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            waveform, sample_rate = sf.read(tmp_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform)) * 0.95
            return {"waveform": waveform, "sample_rate": sample_rate}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _analyze_bpm(self, waveform, sample_rate, duration):
        # Lightweight version for UI, you can swap in your advanced one if needed.
        try:
            tempo = librosa.beat.tempo(y=waveform, sr=sample_rate)
            bpm = float(tempo[0]) if len(tempo) else 120.0
            if bpm > 180: bpm = bpm / 2  # Correct for double-detection
            if bpm < 60: bpm = bpm * 2   # Correct for half-detection
            return np.clip(bpm, 45, 220)
        except Exception:
            return 120.0

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SourceVideoLoader": "🎥 BAIS1C Source Video Loader"}
