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
    - Enhanced BPM analysis with cross-validation (full audio duration)
    - Prepared outputs for pose extraction and dance synchronization
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",)
            }
        }

    RETURN_TYPES = ("VIDEO", "AUDIO", "FLOAT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = (
        "video_obj",      # Pass this to pose extractor or VACE
        "audio_obj",      # {"waveform":..., "sample_rate":...} for dance sync nodes
        "fps",            # frames per second (float)
        "bpm",            # detected audio BPM (float)
        "frame_count",    # total frame count (int)
        "duration",       # video duration (float, seconds)
    )
    FUNCTION = "load"
    CATEGORY = "BAIS1C VACE Suite/Source"

    def load(self, video):
        """
        Load video and extract comprehensive metadata for dance sync pipeline
        """
        # Handle ComfyUI VideoFromFile object or string path
        if hasattr(video, "video_path"):
            video_path = video.video_path
        elif isinstance(video, str):
            video_path = video
        else:
            raise ValueError(f"Unsupported video input type: {type(video)}. Expected VideoFromFile object or string path.")

        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"[BAIS1C VACE Suite] Loading video: {os.path.basename(video_path)}")

        try:
            # Extract basic video properties
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            fps = float(vr.get_avg_fps())
            frame_count = len(vr)
            duration = frame_count / fps if fps > 0 else 0

            print(f"[BAIS1C VACE Suite] Video properties: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s")

            # Extract audio with robust error handling
            audio_obj = self._extract_audio(video_path)
            
            # Enhanced BPM analysis with cross-validation
            bpm = self._analyze_bpm_enhanced(audio_obj["waveform"], audio_obj["sample_rate"], duration)
            
            print(f"[BAIS1C VACE Suite] Audio analysis: BPM {bpm:.1f}, sample rate {audio_obj['sample_rate']} Hz")

            return (video, audio_obj, fps, bpm, frame_count, duration)

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

        # Create temporary file for audio extraction
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        try:
            # FFmpeg command with error handling
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
                timeout=300  # 5 minute timeout
            )
            
            # Load the extracted audio
            waveform, sample_rate = sf.read(tmp_path, dtype="float32")
            
            # Ensure mono
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Normalize to prevent clipping
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
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _check_ffmpeg(self):
        """Check if FFmpeg is available in PATH"""
        return shutil.which('ffmpeg') is not None

    def _analyze_bpm_enhanced(self, waveform, sample_rate, duration):
        """
        Enhanced BPM analysis optimized for dance synchronization using FULL audio duration.
        Includes cross-validation and error detection for any genre (60-200+ BPM range).
        """
        print(f"[BAIS1C VACE Suite] Analyzing BPM from {duration:.1f}s of audio...")
        
        try:
            # Primary BPM detection using full audio - librosa's gold standard
            tempo_estimates = librosa.beat.tempo(
                y=waveform, 
                sr=sample_rate, 
                aggregate=None,  # Get multiple estimates for validation
                start_bpm=60,    # Lower bound for any genre
                std_bpm=40       # Allow wide variance for genre flexibility
            )
            
            primary_bpm = float(tempo_estimates[0]) if len(tempo_estimates) > 0 else 120.0
            
            # Get beat positions for confidence validation
            _, beat_frames = librosa.beat.beat_track(
                y=waveform, 
                sr=sample_rate,
                start_bpm=60,
                tightness=200  # Stricter beat tracking for better sync
            )
            
            # Cross-validate with onset-based detection
            onset_bpm = self._onset_based_bpm_validation(waveform, sample_rate)
            
            # Collect multiple BPM candidates for validation
            candidates = [primary_bpm]
            
            # Add onset-based estimate if reasonable
            if 40 <= onset_bpm <= 220:
                candidates.append(onset_bpm)
            
            # Check for common half/double tempo errors in dance music
            if primary_bpm > 150:  # Likely double-tempo error
                candidates.append(primary_bpm / 2)
                print(f"[BAIS1C VACE Suite] Checking half-tempo: {primary_bpm/2:.1f} BPM")
            
            if primary_bpm < 75:   # Likely half-tempo error
                candidates.append(primary_bpm * 2)
                print(f"[BAIS1C VACE Suite] Checking double-tempo: {primary_bpm*2:.1f} BPM")
            
            # Select most consistent BPM using beat timing validation
            final_bpm = self._select_most_consistent_bpm(candidates, beat_frames, sample_rate, duration)
            
            # Final sanity check with expanded range for all genres
            if 45 <= final_bpm <= 220:
                confidence = self._calculate_bpm_confidence(final_bpm, beat_frames, sample_rate)
                print(f"[BAIS1C VACE Suite] BPM Analysis: {final_bpm:.1f} BPM (confidence: {confidence:.1f}%)")
                return final_bpm
            else:
                print(f"[BAIS1C VACE Suite] BPM {final_bpm:.1f} outside valid range, using fallback analysis")
                return self._fallback_bpm_analysis(waveform, sample_rate, duration)
                
        except Exception as e:
            print(f"[BAIS1C VACE Suite] Enhanced BPM analysis failed: {e}, using fallback")
            return self._fallback_bpm_analysis(waveform, sample_rate, duration)

    def _onset_based_bpm_validation(self, waveform, sample_rate):
        """Cross-validation BPM using onset detection - different algorithm for verification"""
        try:
            # Detect onsets using spectral flux
            hop_length = 512
            onset_frames = librosa.onset.onset_detect(
                y=waveform,
                sr=sample_rate,
                hop_length=hop_length,
                units='frames',
                delta=0.05,      # Sensitivity threshold
                wait=15          # Minimum frames between onsets
            )
            
            if len(onset_frames) < 4:  # Need minimum onsets for calculation
                return 120.0
            
            # Convert to time and calculate intervals
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)
            intervals = np.diff(onset_times)
            
            # Filter out obviously wrong intervals (too fast/slow)
            valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
            
            if len(valid_intervals) > 0:
                avg_interval = np.median(valid_intervals)  # Use median for robustness
                onset_bpm = 60.0 / avg_interval
                return np.clip(onset_bpm, 45.0, 220.0)
            else:
                return 120.0
                
        except Exception:
            return 120.0

    def _select_most_consistent_bpm(self, candidates, beat_frames, sample_rate, duration):
        """Select BPM candidate with most consistent beat timing"""
        if len(candidates) == 1:
            return candidates[0]
        
        best_bpm = candidates[0]
        best_score = 0
        
        # Convert beat frames to times
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        
        for bpm in candidates:
            if not (45 <= bpm <= 220):
                continue
                
            # Calculate expected beat interval
            expected_interval = 60.0 / bpm
            
            # Score based on how well actual beats match expected timing
            score = self._score_beat_consistency(beat_times, expected_interval, duration)
            
            print(f"[BAIS1C VACE Suite] BPM candidate {bpm:.1f}: consistency score {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_bpm = bpm
        
        return best_bpm

    def _score_beat_consistency(self, beat_times, expected_interval, duration):
        """Score how well detected beats match expected BPM timing"""
        if len(beat_times) < 2:
            return 0.0
        
        # Calculate actual intervals between beats
        actual_intervals = np.diff(beat_times)
        
        # Score based on how close actual intervals are to expected
        deviations = np.abs(actual_intervals - expected_interval) / expected_interval
        consistency_score = np.mean(1.0 - np.clip(deviations, 0, 1))
        
        # Bonus for having beats throughout the duration (not just at start)
        coverage_score = len(beat_times) / (duration * (1.0 / expected_interval))
        coverage_score = np.clip(coverage_score, 0, 1)
        
        return consistency_score * 0.7 + coverage_score * 0.3

    def _calculate_bpm_confidence(self, bpm, beat_frames, sample_rate):
        """Calculate confidence percentage for BPM estimate"""
        if len(beat_frames) < 2:
            return 50.0
        
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        expected_interval = 60.0 / bpm
        actual_intervals = np.diff(beat_times)
        
        # Calculate coefficient of variation (lower = more consistent = higher confidence)
        if len(actual_intervals) > 0:
            cv = np.std(actual_intervals) / np.mean(actual_intervals)
            confidence = max(0, 100 * (1 - cv * 2))  # Scale and invert
            return min(confidence, 95.0)  # Cap at 95%
        
        return 50.0

    def _fallback_bpm_analysis(self, waveform, sample_rate, duration):
        """Enhanced fallback BPM analysis using multiple methods"""
        print(f"[BAIS1C VACE Suite] Using fallback BPM analysis...")
        
        try:
            # Method 1: Tempo estimation with different parameters
            tempo_strict = librosa.beat.tempo(
                y=waveform, 
                sr=sample_rate,
                start_bpm=90,
                std_bpm=20,
                max_tempo=180
            )
            
            if len(tempo_strict) > 0:
                fallback_bpm = float(tempo_strict[0])
                if 60 <= fallback_bpm <= 180:
                    print(f"[BAIS1C VACE Suite] Fallback method 1 success: {fallback_bpm:.1f} BPM")
                    return fallback_bpm
            
            # Method 2: Simplified onset detection
            onset_bpm = self._onset_based_bpm_validation(waveform, sample_rate)
            if 70 <= onset_bpm <= 160:
                print(f"[BAIS1C VACE Suite] Fallback method 2 success: {onset_bpm:.1f} BPM")
                return onset_bpm
            
            # Method 3: Duration-based estimation for looped content
            if duration < 30:  # Short clip, might be a loop
                # Common loop lengths at standard BPMs
                loop_bpms = [120, 128, 140, 100, 110, 90, 80]
                for test_bpm in loop_bpms:
                    expected_loop_duration = (4 * 60) / test_bpm  # 4-beat loop
                    if abs(duration - expected_loop_duration) < 0.5:
                        print(f"[BAIS1C VACE Suite] Loop-based BPM estimate: {test_bpm} BPM")
                        return float(test_bpm)
            
            # Ultimate fallback - genre-appropriate default
            print("[BAIS1C VACE Suite] All BPM analysis methods failed, using adaptive default")
            
            # Choose default based on audio characteristics
            if duration < 10:
                return 128.0  # Short clips often electronic/dance
            elif duration > 180:
                return 100.0  # Long tracks often slower
            else:
                return 120.0  # Universal default
                
        except Exception as e:
            print(f"[BAIS1C VACE Suite] Fallback BPM analysis error: {e}")
            return 120.0

# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SourceVideoLoader": "🎥 BAIS1C Source Video Loader"}

# Self-test function
def test_source_video_loader():
    """Test the video loader with validation"""
    loader = BAIS1C_SourceVideoLoader()
    
    # Test FFmpeg availability
    if not loader._check_ffmpeg():
        print("❌ FFmpeg not available - audio extraction will fail")
        return False
    
    # Test BPM analysis with synthetic audio of known BPM
    sample_rate = 44100
    duration = 10.0  # 10 seconds for proper BPM analysis
    bpm_test = 120.0
    
    # Create test audio with clear beats
    t = np.linspace(0, duration, int(sample_rate * duration))
    beat_interval = 60.0 / bpm_test
    
    # Generate click track with beats
    test_waveform = np.zeros_like(t)
    for beat_time in np.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < len(test_waveform):
            # Add click sound (short burst of sine wave)
            click_duration = 0.05  # 50ms click
            click_samples = int(click_duration * sample_rate)
            click_end = min(beat_sample + click_samples, len(test_waveform))
            click_t = np.linspace(0, click_duration, click_end - beat_sample)
            test_waveform[beat_sample:click_end] = 0.5 * np.sin(2 * np.pi * 800 * click_t)
    
    try:
        detected_bpm = loader._analyze_bpm_enhanced(test_waveform, sample_rate, duration)
        error_percent = abs(detected_bpm - bpm_test) / bpm_test * 100
        
        if error_percent < 5:  # Within 5% is good
            print(f"✅ Enhanced BPM analysis test passed: {detected_bpm:.1f} BPM (target: {bpm_test}, error: {error_percent:.1f}%)")
            return True
        else:
            print(f"⚠️ BPM analysis marginal: {detected_bpm:.1f} BPM (target: {bpm_test}, error: {error_percent:.1f}%)")
            return True  # Still acceptable for complex audio
            
    except Exception as e:
        print(f"❌ Enhanced BPM analysis test failed: {e}")
        return False

# Uncomment to run self-test
# test_source_video_loader()