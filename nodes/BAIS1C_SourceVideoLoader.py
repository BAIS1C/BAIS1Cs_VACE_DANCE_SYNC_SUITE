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
    # Try vars() - but filter out private attributes
    if hasattr(input_obj, "__dict__"):
        obj_dict = vars(input_obj)
        # Filter out private/protected attributes and methods
        filtered_dict = {k: v for k, v in obj_dict.items() 
                        if not k.startswith('_') and not callable(v)}
        return filtered_dict
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
        """Perform comprehensive audio analysis with consistent frame sizing"""
        if not audio or "waveform" not in audio or "sample_rate" not in audio:
            return None
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Handle tensor input
        if hasattr(waveform, 'detach'):
            waveform = waveform.detach().cpu().numpy()
        
        try:
            analyzer = EnhancedAudioAnalyzer(sample_rate)
            analysis = analyzer.analyze_comprehensive(waveform, target_fps=target_fps, debug=False)
            
            # Verify frame consistency
            expected_frames = analysis.get('total_frames', 0)
            if expected_frames > 0:
                # Check key frame-based arrays
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
        """Main execution with consistent sync_meta handling"""
        # Convert video_info to sync_meta consistently
        sync_meta = ensure_dict(video_info)
        
        # Extract or default key parameters
        target_fps = float(sync_meta.get("loaded_fps", sync_meta.get("source_fps", sync_meta.get("fps", 24.0))))
        duration = float(sync_meta.get("loaded_duration", sync_meta.get("source_duration", sync_meta.get("duration", 1.0))))
        frame_count = int(sync_meta.get("loaded_frame_count", sync_meta.get("source_frame_count", sync_meta.get("frame_count", 24))))
        
        # Add standardized fields to sync_meta
        sync_meta.update({
            "fps": target_fps,
            "duration": duration,
            "frame_count": frame_count,
            "audio_present": audio is not None and "waveform" in audio
        })
        
        # Perform audio analysis if available
        audio_analysis = None
        if audio and audio.get("waveform") is not None and audio.get("sample_rate") is not None:
            audio_analysis = self.analyze_audio(audio, target_fps)
            if audio_analysis:
                # Merge audio analysis into sync_meta
                sync_meta.update(audio_analysis)
                sync_meta["audio_analysis_performed"] = True
                
                # Verify critical fields are present
                required_fields = ['primary_bpm', 'total_frames', 'beat_strength', 'freq_bands']
                missing_fields = [field for field in required_fields if field not in sync_meta]
                if missing_fields:
                    print(f"[SourceVideoLoader] WARNING: Missing audio analysis fields: {missing_fields}")
            else:
                sync_meta["audio_analysis_performed"] = False
                print("[SourceVideoLoader] Audio analysis failed - using defaults")
                # Add minimal defaults to prevent downstream errors
                sync_meta.update({
                    "primary_bpm": 120.0,
                    "bpm_confidence": 0.0,
                    "beat_strength": np.zeros(frame_count).tolist(),
                    "beat_times": [],
                    "freq_bands": {
                        "bass": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0},
                        "mid": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0},
                        "highs": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0}
                    }
                })
        else:
            sync_meta["audio_analysis_performed"] = False
            print("[SourceVideoLoader] No audio provided - using defaults")
            # Add minimal defaults
            sync_meta.update({
                "primary_bpm": 120.0,
                "bpm_confidence": 0.0,
                "beat_strength": np.zeros(frame_count).tolist(),
                "beat_times": [],
                "freq_bands": {
                    "bass": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0},
                    "mid": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0},
                    "highs": {"energy": np.zeros(frame_count).tolist(), "rms": 0.0}
                }
            })

        # Generate UI info summary
        ui_info = self._create_ui_info(sync_meta, audio_analysis)
        
        return (images, audio, sync_meta, ui_info)

    def _create_ui_info(self, sync_meta, audio_analysis):
        """Create comprehensive UI info string"""
        lines = []
        
        # Basic video info
        lines.append(f"📺 Video: {sync_meta.get('frame_count', 0)} frames @ {sync_meta.get('fps', 0):.1f} FPS")
        lines.append(f"⏱️  Duration: {sync_meta.get('duration', 0):.2f}s")
        
        # Audio analysis results
        if audio_analysis:
            lines.append(f"🎵 BPM: {audio_analysis.get('primary_bpm', 0):.2f} (Confidence: {audio_analysis.get('bpm_confidence', 0):.2f})")
            lines.append(f"🥁 Beat Consistency: {audio_analysis.get('beat_consistency', 0):.2f}")
            lines.append(f"🎼 Total Frames: {audio_analysis.get('total_frames', 0)}")
            
            # Rhythm characteristics
            if "rhythm_patterns" in audio_analysis:
                rhythm = audio_analysis["rhythm_patterns"]
                swing_status = "Swing" if rhythm.get("has_swing", False) else "Straight"
                lines.append(f"🎶 Rhythm: {swing_status} (Ratio: {rhythm.get('swing_ratio', 1.0):.2f})")
                lines.append(f"🎯 Groove Strength: {rhythm.get('groove_strength', 0):.2f}")
            
            # Frequency analysis summary
            if "freq_bands" in audio_analysis:
                bands = audio_analysis["freq_bands"]
                band_summary = []
                for band_name in ["bass", "mid", "highs"]:
                    if band_name in bands:
                        rms = bands[band_name].get("rms", 0)
                        band_summary.append(f"{band_name.title()}: {rms:.3f}")
                if band_summary:
                    lines.append(f"🔊 Frequency RMS: {' | '.join(band_summary)}")
            
            # Beat detection info
            beat_count = len(audio_analysis.get('beat_times', []))
            lines.append(f"🎵 Detected Beats: {beat_count}")
            
        else:
            lines.append("🔇 No audio analysis performed")
            lines.append("   Using default values for synchronization")
        
        # Frame consistency check
        expected_frames = sync_meta.get('total_frames', sync_meta.get('frame_count', 0))
        if audio_analysis and expected_frames > 0:
            beat_strength_len = len(sync_meta.get('beat_strength', []))
            if beat_strength_len == expected_frames:
                lines.append(f"✅ Frame consistency: {expected_frames} frames")
            else:
                lines.append(f"⚠️  Frame mismatch: {beat_strength_len} vs {expected_frames}")
        
        return "\n".join(lines)

# ComfyUI Node registration block
NODE_CLASS_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": BAIS1C_SourceVideoLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_SourceVideoLoader": "BAIS1C Source Video Loader (BPM/Meta)",
}

# Self-test function
def _test_source_video_loader():
    """Test SourceVideoLoader with various input types"""
    print("🔍 Testing BAIS1C_SourceVideoLoader...")
    
    # Test data
    sr = 44100
    duration = 2.0
    test_waveform = np.sin(2 * np.pi * 2 * np.linspace(0, duration, int(sr * duration)))
    test_audio = {"waveform": test_waveform, "sample_rate": sr}
    test_images = np.random.randint(0, 255, (48, 480, 640, 3), dtype=np.uint8)  # 2s @ 24fps
    
    # Test with dict input
    test_meta_dict = {"fps": 24.0, "duration": duration, "frame_count": 48}
    
    # Test with object-like input
    class DummyInfo:
        def __init__(self):
            self.loaded_fps = 24.0
            self.loaded_duration = duration
            self.loaded_frame_count = 48
        def to_dict(self):
            return {"loaded_fps": self.loaded_fps, "loaded_duration": self.loaded_duration, "loaded_frame_count": self.loaded_frame_count}
    
    test_meta_obj = DummyInfo()
    
    node = BAIS1C_SourceVideoLoader()
    
    try:
        # Test 1: Dict input
        images, audio, sync_meta, ui_info = node.execute(test_images, test_audio, test_meta_dict)
        
        assert isinstance(sync_meta, dict), "sync_meta should be dict"
        assert "primary_bpm" in sync_meta, "Should contain BPM analysis"
        assert "fps" in sync_meta, "Should contain fps"
        assert sync_meta["audio_analysis_performed"] == True, "Should indicate audio analysis was performed"
        
        print("✅ Dict input test passed")
        print(f"   BPM: {sync_meta['primary_bpm']:.1f}")
        print(f"   Frames: {sync_meta['total_frames']}")
        
        # Test 2: Object input
        images2, audio2, sync_meta2, ui_info2 = node.execute(test_images, test_audio, test_meta_obj)
        
        assert isinstance(sync_meta2, dict), "sync_meta should be dict"
        assert "primary_bpm" in sync_meta2, "Should contain BPM analysis"
        
        print("✅ Object input test passed")
        
        # Test 3: No audio
        images3, audio3, sync_meta3, ui_info3 = node.execute(test_images, None, test_meta_dict)
        
        assert sync_meta3["audio_analysis_performed"] == False, "Should indicate no audio analysis"
        assert "primary_bpm" in sync_meta3, "Should have default BPM"
        assert sync_meta3["primary_bpm"] == 120.0, "Should use default BPM"
        
        print("✅ No audio test passed")
        
        print("✅ All SourceVideoLoader tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ SourceVideoLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Inline test function (for standalone/CLI dev testing)
if __name__ == "__main__":
    _test_source_video_loader()