import torch, numpy as np, os, json, glob, librosa

class BAIS1C_MusicControlNet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pose_tensor": ("TENSOR",),
                "sync_meta": ("DICT",),
                "library_dance": (cls._get_available_dances(), {"default": "none"}),
                "reactivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "report": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("TENSOR", "DICT", "STRING")
    RETURN_NAMES = ("modulated_pose", "sync_meta", "report_str")
    FUNCTION = "modulate"
    CATEGORY = "BAIS1C VACE Suite/Control"

    @classmethod
    def _get_available_dances(cls):
        suite_dir = os.path.dirname(__file__)
        library_dir = os.path.abspath(os.path.join(suite_dir, "../dance_library"))
        dances = ["none"]
        if not os.path.exists(library_dir):
            return dances
        for f in glob.glob(os.path.join(library_dir, "*.json")):
            try:
                base = os.path.splitext(os.path.basename(f))[0]
                dances.append(base)
            except Exception:
                pass
        return dances

    def _load_pose_from_library(self, library_dance, debug=False):
        if library_dance == "none":
            return None
        suite_dir = os.path.dirname(__file__)
        library_dir = os.path.abspath(os.path.join(suite_dir, "../dance_library"))
        json_path = os.path.join(library_dir, f"{library_dance}.json")
        if not os.path.exists(json_path):
            if debug:
                print(f"[MusicControlNet] JSON not found: {json_path}")
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "pose_tensor" in data:
                return torch.FloatTensor(data["pose_tensor"])
        except Exception as e:
            if debug:
                print(f"[MusicControlNet] Failed to load {library_dance}: {e}")
        return None

    def modulate(self, audio, pose_tensor, sync_meta, library_dance, reactivity, report):
        debug = report
        # Try to load pose sequence from library, if requested
        library_pose = self._load_pose_from_library(library_dance, debug=debug)
        if library_pose is not None:
            if debug:
                print(f"[MusicControlNet] Loaded {library_dance} from library ({library_pose.shape})")
            pose_tensor = library_pose
            library_loaded = True
        else:
            if debug and library_dance != "none":
                print(f"[MusicControlNet] Library {library_dance} not found, using input pose tensor.")
            library_loaded = False

        waveform, sample_rate = audio.get("waveform"), audio.get("sample_rate")
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        duration = len(waveform) / sample_rate
        n_frames = pose_tensor.shape[0]
        audio_features = self._analyze_audio(waveform, sample_rate, n_frames)

        pose_np = pose_tensor.cpu().numpy() if isinstance(pose_tensor, torch.Tensor) else np.array(pose_tensor)
        modulated = pose_np.copy()
        for i in range(n_frames):
            if audio_features["beat_strength"][i] > 0.5:
                modulated[i, :, 1] += reactivity * 0.01

        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "modulation": "music_reactivity",
            "audio_reactivity": reactivity,
            "library_dance": library_dance,
            "library_loaded": library_loaded,
        })
        report_str = ""
        if report:
            report_str = (
                f"Music Control Net Report\n"
                f"Frames: {n_frames}, Duration: {duration:.2f}s, "
                f"BPM: {audio_features['bpm']:.2f}\n"
                f"Reactivity: {reactivity:.2f}\n"
                f"Library Dance: {library_dance} (loaded: {library_loaded})\n"
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
    out = node.modulate(audio, pose_tensor, sync_meta, "none", 0.5, True)
    print(out[2])
# _test_music_control_net()
