import os
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

    Loads video files and extracts comprehensive metadata including:
    - Video properties (FPS, frame count, duration)
    - Enhanced BPM analysis (full audio duration)
    - Outputs for pose extraction and dance synchronization
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",)
            }
        }

    RETURN_TYPES = ("VIDEO", "AUDIO", "DICT", "STRING")
    RETURN_NAMES = (
        "video_obj",    # For pose extractor or downstream nodes
        "audio_obj",    # {"waveform":..., "sample_rate":...}
        "sync_meta",    # All meta info as a dict
        "ui_info",      # Info string for ComfyUI UI
    )
    FUNCTION = "load"
    CATEGORY = "BAIS1C VACE Suite/Source"

    def load(self, video):
        """
        Load video and extract comprehensive metadata for dance sync pipeline.
        Accepts ComfyUI video types, VideoFromFile, string path, etc.
        """
        # --- Robust video path extraction ---
        video_path = None
        debug_info = []

        if isinstance(video, str):
            video_path = video
            debug_info.append("input is str")
        elif hasattr(video, "video_path"):
            video_path = getattr(video, "video_path")
            debug_info.append("input has .video_path")
        elif hasattr(video, "filename"):
            video_path = getattr(video, "filename")
            debug_info.append("input has .filename")
        elif hasattr(video, "path"):
            video_path = getattr(video, "path")
            debug_info.append("input has .path")
        elif hasattr(video, "__dict__"):
            if "video_path" in video.__dict__:
                video_path = video.__dict__["video_path"]
                debug_info.append("input __dict__['video_path']")
            elif "filename" in video.__dict__:
                video_path = video.__dict__["filename"]
                debug_info.append("input __dict__['filename']")
            elif "path" in video.__dict__:
                video_path = video.__dict__["path"]
                debug_info.append("input __dict__['path']")
            elif "_VideoFromFile__file" in video.__dict__:
                video_path = video.__dict__["_VideoFromFile__file"]
                debug_info.append("input __dict__['_VideoFromFile__file']")

        if not video_path or not isinstance(video_path, str):
            print(f"[BAIS1C SourceVideoLoader] DEBUG: video input type {type(video)} dir={dir(video)} __dict__={getattr(video, '__dict__', None)}")
            raise ValueError(f"Unsupported video input type: {type(video)}. Object has no path attribute.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"[BAIS1C VACE Suite] Loading video: {os.path.basename(video_path)} ({', '.join(debug_info)})")

        try:
            # --- Video properties ---
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            fps = float(vr.get_avg_fps())
            frame_count = len(vr)
            duration = frame_count / fps if fps > 0 else 0

            print(f"[BAIS1C VACE Suite] Video properties: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s")

            # --- Audio extraction ---
            audio_obj = self._extract_audio(video_path)

            # --- Enhanced BPM analysis ---
            bpm = self._analyze_bpm_enhanced(audio_obj["waveform"], audio_obj["sample_rate"], duration)

            print(f"[BAIS1C VACE Suite] Audio analysis: BPM {bpm:.1f}, sample rate {audio_obj['sample_rate']} Hz")

            # --- Sync meta ---
            sync_meta = {
                "video_path": video_path,
                "video_name": os.path.basename(video_path),
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "bpm": bpm,
                "sample_rate": audio_obj["sample_rate"],
            }

            ui_info = (
                f"{os.path.basename(video_path)}\n"
                f"{frame_count} frames @ {fps:.2f} FPS\n"
                f"Duration: {duration:.2f} s\n"
                f"BPM: {bpm:.1f}\n"
                f"Sample Rate: {audio_obj['sample_rate']}\n"
            )

            return (video, audio_obj, sync_meta, ui_info)

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

    def _extract_audio(self, video_path):
        """Extract audio from video using FFmpeg with robust error handling"""
        # Check if FFmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError(
                "FFmpeg not found in PATH. Please install FFmpeg to extract audio from videos.\n"
                "Download from: https://ffmpeg.org/download.html"
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '44100',  # 44.1kHz sample rate
                '-ac', '1',  # Mono
                tmp_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )

            waveform, sample_rate = sf.read(tmp_path, dtype="float32")

            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform)) * 0.95

            return {"waveform": waveform, "sample_rate": sample_rate}

        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out (5 minutes). Video file may be too large or corrupted.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed to extract audio: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _check_ffmpeg(self):
        """Check if FFmpeg is available in PATH"""
        return shutil.which('ffmpeg') is not None

    def _analyze_bpm_enhanced(self, waveform, sample_rate, duration):
    print(f"[BAIS1C VACE Suite] Analyzing BPM from {duration:.1f}s of audio...")
    try:
        # Get all tempo estimates
        tempo_estimates = librosa.beat.tempo(
            y=waveform,
            sr=sample_rate,
            aggregate=None,
            start_bpm=60,
            std_bpm=40
        )
        # Filter plausible "song" BPMs (real-world music)
        plausible_bpms = [b for b in tempo_estimates if 70 <= b <= 160]
        if plausible_bpms:
            primary_bpm = float(min(plausible_bpms))
        else:
            primary_bpm = float(np.median(tempo_estimates)) if len(tempo_estimates) > 0 else 120.0
        # Halve anything above 150 BPM, recursively (catches quadruple tempo)
        while primary_bpm > 150:
            primary_bpm /= 2.0
        return np.clip(primary_bpm, 60, 150)
    except Exception as e:
        print(f"[BAIS1C VACE Suite] BPM analysis failed: {e}, using default 120.0")
        return 120.0


# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SourceVideoLoader": "🎥 BAIS1C Source Video Loader"}

# Optional: Self-test (for direct script execution)
def test_source_video_loader():
    """Test with a path string and dummy video-like objects."""
    class DummyVideo:
        def __init__(self, path):
            self.video_path = path
    test_path = "YOUR_TEST_VIDEO.mp4"
    loader = BAIS1C_SourceVideoLoader()
    for inp in [test_path, DummyVideo(test_path)]:
        try:
            loader.load(inp)
            print(f"Passed: {type(inp)}")
        except Exception as e:
            print(f"Failed: {type(inp)}: {e}")
# test_source_video_loader()
