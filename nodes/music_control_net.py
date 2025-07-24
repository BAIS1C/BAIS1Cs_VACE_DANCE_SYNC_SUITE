# music_control_net.py
import torch, numpy as np
import librosa

class BAIS1C_MusicControlNet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pose_tensor": ("TENSOR",),
                "sync_meta": ("DICT",),  # always sync_meta!
                "reactivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "report": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("TENSOR", "DICT", "STRING")
    RETURN_NAMES = ("modulated_pose", "sync_meta", "report_str")
    FUNCTION = "modulate"
    CATEGORY = "BAIS1C VACE Suite/Control"

    def modulate(self, audio, pose_tensor, sync_meta, reactivity, report):
        # Analyze audio for basic features
        waveform, sample_rate = audio.get("waveform"), audio.get("sample_rate")
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        duration = len(waveform) / sample_rate
        n_frames = pose_tensor.shape[0]
        audio_features = self._analyze_audio(waveform, sample_rate, n_frames)
        
        # Very simple "reactivity" modulation: add vertical motion on beat
        pose_np = pose_tensor.cpu().numpy() if isinstance(pose_tensor, torch.Tensor) else np.array(pose_tensor)
        modulated = pose_np.copy()
        for i in range(n_frames):
            if audio_features["beat_strength"][i] > 0.5:
                modulated[i, :, 1] += reactivity * 0.01  # vertical "pop" for all joints

        # Prepare sync_meta propagation
        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "modulation": "music_reactivity",
            "audio_reactivity": reactivity
        })
        report_str = ""
        if report:
            report_str = (
                f"Music Control Net Report\n"
                f"Frames: {n_frames}, Duration: {duration:.2f}s, "
                f"BPM: {audio_features['bpm']:.2f}\n"
                f"Reactivity: {reactivity:.2f}\n"
                f"Input Meta: {sync_meta_out}\n"
            )
        return torch.from_numpy(modulated), sync_meta_out, report_str

    def _analyze_audio(self, waveform, sample_rate, n_frames):
        try:
            tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
            bpm = float(tempo)
            _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
            beat_strength = np.zeros(n_frames)
            duration = len(waveform) / sample_rate
            for bt in beat_times:
                idx = int((bt / duration) * n_frames)
                if 0 <= idx < n_frames:
                    beat_strength[idx] = 1.0
        except Exception:
            bpm = 120.0
            beat_strength = np.zeros(n_frames)
        return {"bpm": bpm, "beat_strength": beat_strength}

NODE_CLASS_MAPPINGS = {"BAIS1C_MusicControlNet": BAIS1C_MusicControlNet}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_MusicControlNet": "🎵 BAIS1C Music Control Net"}

# -- Self-test --
def _test_music_control_net():
    node = BAIS1C_MusicControlNet()
    pose_tensor = torch.zeros((8,128,2))
    sync_meta = {"fps":24,"title":"test"}
    audio = {"waveform":np.random.randn(24*2*8),"sample_rate":48_000}
    out = node.modulate(audio, pose_tensor, sync_meta, 0.5, True)
    print(out[2])
# _test_music_control_net()
