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
            return None, None
        suite_dir = os.path.dirname(__file__)
        library_dir = os.path.abspath(os.path.join(suite_dir, "../dance_library"))
        json_path = os.path.join(library_dir, f"{library_dance}.json")
        if not os.path.exists(json_path):
            if debug:
                print(f"[MusicControlNet] JSON not found: {json_path}")
            return None, None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bpm = data.get("metadata", {}).get("bpm", 120.0)
            if "pose_tensor" in data:
                return torch.FloatTensor(data["pose_tensor"]), bpm
        except Exception as e:
            if debug:
                print(f"[MusicControlNet] Failed to load {library_dance}: {e}")
        return None, None

    def _analyze_audio(self, waveform, sample_rate, n_frames):
        try:
            tempo = librosa.beat.tempo(y=waveform, sr=sample_rate)
            bpm = float(tempo[0]) if hasattr(tempo, '__getitem__') else float(tempo)
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

    def _bpm_sync_pose_tensor(self, pose_tensor, orig_bpm, target_bpm, debug=False):
        """
        Resample a pose tensor so that its tempo matches the new target BPM.
        Args:
            pose_tensor: torch.Tensor (frames, kpts, dims) or np.ndarray
            orig_bpm: original movement BPM (float)
            target_bpm: new music BPM (float)
        Returns:
            retimed_pose: torch.Tensor, new length in frames
        """
        if not isinstance(pose_tensor, np.ndarray):
            pose_np = pose_tensor.cpu().numpy()
        else:
            pose_np = pose_tensor

        n_orig = pose_np.shape[0]
        factor = float(target_bpm) / float(orig_bpm)
        n_target = int(np.round(n_orig / factor))
        if n_target < 2:
            n_target = 2
        if debug:
            print(f"[bpm_sync] Orig frames: {n_orig}, Orig BPM: {orig_bpm}, Target BPM: {target_bpm}, Target frames: {n_target}, Factor: {factor:.3f}")
        old_idx = np.linspace(0, n_orig-1, n_orig)
        new_idx = np.linspace(0, n_orig-1, n_target)
        out = np.stack([
            np.stack([np.interp(new_idx, old_idx, pose_np[:,k,d]) for d in range(pose_np.shape[2])], axis=1)
            for k in range(pose_np.shape[1])
        ], axis=1).transpose(1,0,2)
        return torch.from_numpy(out.astype(np.float32))

    def modulate(self, audio, pose_tensor, sync_meta, library_dance, reactivity, report):
        debug = report
        # --- Library pose loading (with meta bpm) ---
        library_loaded = False
        orig_bpm = sync_meta.get("bpm", 120.0)
        if library_dance != "none":
            lib_pose, lib_bpm = self._load_pose_from_library(library_dance, debug=debug)
            if lib_pose is not None:
                pose_tensor = lib_pose
                orig_bpm = lib_bpm if lib_bpm is not None else orig_bpm
                library_loaded = True
                if debug:
                    print(f"[MusicControlNet] Loaded {library_dance} from library ({pose_tensor.shape[0]} frames, bpm={orig_bpm})")
            else:
                if debug:
                    print(f"[MusicControlNet] Library {library_dance} not found, using input pose tensor.")

        # --- Analyze audio and get new target BPM ---
        waveform, sample_rate = audio.get("waveform"), audio.get("sample_rate")
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        n_input_frames = pose_tensor.shape[0]
        audio_features = self._analyze_audio(waveform, sample_rate, n_input_frames)
        target_bpm = audio_features["bpm"]

        # --- BPM sync: retime pose sequence to match new music BPM ---
        pose_tensor_synced = self._bpm_sync_pose_tensor(pose_tensor, orig_bpm, target_bpm, debug=debug)

        # --- Beat reactivity modulation (optional) ---
        n_frames = pose_tensor_synced.shape[0]
        audio_features = self._analyze_audio(waveform, sample_rate, n_frames)
        pose_np = pose_tensor_synced.cpu().numpy()
        modulated = pose_np.copy()
        for i in range(n_frames):
            if audio_features["beat_strength"][i] > 0.5:
                modulated[i, :, 1] += reactivity * 0.01  # simple vertical "pop"

        # --- Propagate updated meta ---
        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "modulation": "music_bpm_sync+reactivity",
            "audio_reactivity": reactivity,
            "library_dance": library_dance,
            "library_loaded": library_loaded,
            "orig_bpm": orig_bpm,
            "target_bpm": target_bpm,
            "pose_frames": pose_tensor.shape[0],
            "synced_frames": n_frames
        })
        report_str = ""
        if report:
            report_str = (
                f"Music Control Net Report\n"
                f"Library: {library_dance} (loaded: {library_loaded})\n"
                f"Original BPM: {orig_bpm}, Target BPM: {target_bpm}\n"
                f"Orig Frames: {pose_tensor.shape[0]}, Synced Frames: {n_frames}\n"
                f"Reactivity: {reactivity:.2f}\n"
                f"Meta: {sync_meta_out}\n"
            )
        return torch.from_numpy(modulated), sync_meta_out, report_str

NODE_CLASS_MAPPINGS = {"BAIS1C_MusicControlNet": BAIS1C_MusicControlNet}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_MusicControlNet": "🎵 BAIS1C Music Control Net"}

# -- Self-test --
def _test_music_control_net():
    node = BAIS1C_MusicControlNet()
    pose_tensor = torch.zeros((100,128,2))
    sync_meta = {"fps":24,"title":"test", "bpm":100}
    audio = {"waveform":np.random.randn(48000*5),"sample_rate":48000}
    out = node.modulate(audio, pose_tensor, sync_meta, "none", 0.5, True)
    print(out[2])
# _test_music_control_net()
