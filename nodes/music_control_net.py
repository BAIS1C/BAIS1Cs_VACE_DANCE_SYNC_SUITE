# music_control_net.py  (BAIS1C VACE Dance Sync Suite – Auto-Sync Edition)
import torch
import numpy as np
import librosa
import json
import os
import glob
import cv2
from typing import List, Dict, Tuple, Optional

class BAIS1C_MusicControlNet:
    """
    BAIS1C VACE Dance Sync Suite – Music Control Net (Auto-Sync)
    • Auto-BPM from audio
    • Auto-length to match audio
    • Accepts direct pose tensor OR library JSON
    • Wan-safe aspect & resolution caps
    """

    # ---------- Wan-safe limits ----------
    MAX_W = 460
    MAX_H = 832
    VALID_RATIOS = {16/9, 9/16}

    @classmethod
    def INPUT_TYPES(cls):
        dances = cls._get_available_dances()
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 12.0, "max": 60.0}),

                "pose_source": (["direct_tensor", "library_json"], {"default": "direct_tensor"}),
                "library_dance": (dances, {"default": "none"}),

                "sync_method": (["time_domain", "frame_perfect", "beat_aligned"], {"default": "frame_perfect"}),
                "loop_mode": (["once", "loop_to_fit", "crop_to_fit"], {"default": "loop_to_fit"}),

                "generate_video": ("BOOLEAN", {"default": True}),
                "video_style": (["stickman", "dots", "skeleton"], {"default": "stickman"}),
                "save_synced_json": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "auto_synced"}),
                "smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_pose_tensor": ("POSE",),
            }
        }

    RETURN_TYPES = ("POSE", "IMAGE", "STRING")
    RETURN_NAMES = ("synced_pose_tensor", "pose_video", "sync_report")
    FUNCTION = "sync_pose_to_music"
    CATEGORY = "BAIS1C VACE Suite/Control"

    # ---------- Library scan ----------
    @classmethod
    def _get_available_dances(cls) -> List[str]:
        try:
            suite_dir = None
            check_dir = os.path.dirname(os.path.abspath(__file__))
            for _ in range(5):
                if os.path.exists(os.path.join(check_dir, "dance_library")):
                    suite_dir = check_dir
                    break
                parent = os.path.dirname(check_dir)
                if parent == check_dir:
                    break
                check_dir = parent
            if not suite_dir:
                return ["none"]
            lib = os.path.join(suite_dir, "dance_library")
            files = glob.glob(os.path.join(lib, "*.json"))
            names = ["none"] + [os.path.splitext(os.path.basename(f))[0] for f in files]
            return names
        except Exception:
            return ["none"]

    # ---------- Auto-BPM ----------
    def _extract_bpm(self, audio: dict) -> float:
        try:
            return float(librosa.beat.tempo(y=audio["waveform"], sr=audio["sample_rate"], aggregate=None)[0])
        except Exception:
            return 120.0

    # ---------- Wan-safe dimensions ----------
    def _clamp_res(self, w: int, h: int) -> Tuple[int, int]:
        # enforce 16:9 or 9:16
        ratio = w / h
        if abs(ratio - 16/9) < abs(ratio - 9/16):
            target_ratio = 16/9
        else:
            target_ratio = 9/16

        # scale to fit inside max box
        if w > h:  # landscape
            w_new = min(w, self.MAX_W)
            h_new = int(w_new / target_ratio)
            if h_new > self.MAX_H:
                h_new = self.MAX_H
                w_new = int(h_new * target_ratio)
        else:      # portrait
            h_new = min(h, self.MAX_H)
            w_new = int(h_new / target_ratio)
            if w_new > self.MAX_W:
                w_new = self.MAX_W
                h_new = int(w_new * target_ratio)

        return max(2, w_new), max(2, h_new)

    # ---------- Main sync ----------
    def sync_pose_to_music(self, audio, target_fps, pose_source, library_dance,
                           sync_method, loop_mode, generate_video, video_style,
                           save_synced_json, output_filename, smoothing, debug,
                           input_pose_tensor=None):

        if debug:
            print("\n[BAIS1C MusicControlNet] === Auto-Sync Start ===")

        # 1. Auto-BPM & length
        target_bpm = self._extract_bpm(audio)
        audio_duration = len(audio["waveform"]) / audio["sample_rate"]
        target_frames = int(audio_duration * target_fps)

        # 2. Load pose
        if pose_source == "direct_tensor" and input_pose_tensor is not None:
            pose_data = input_pose_tensor.cpu().numpy()
            source_metadata = {"source_bpm": 120, "source_fps": 24, "title": "direct"}
        elif pose_source == "library_json" and library_dance != "none":
            pose_data, source_metadata = self._load_from_json_library(library_dance, debug)
        else:
            empty = torch.zeros((1, 128, 2))
            empty_img = torch.zeros((1, self.MAX_H, self.MAX_W, 3))
            return empty, empty_img, "❌ No pose data"

        # 3. Sync transform
        synced_poses = self._resample_and_sync(
            pose_data, source_metadata, target_bpm, target_fps, sync_method, loop_mode, smoothing, target_frames, debug
        )

        # 4. Wan-safe video
        w, h = self._clamp_res(512, 896)  # defaults
        pose_video = None
        if generate_video:
            pose_video = self._generate_pose_video(synced_poses, w, h, "black", video_style, target_fps, debug)
        else:
            pose_video = torch.zeros((1, h, w, 3))

        # 5. Optional save
        if save_synced_json:
            self._save_synced_json(synced_poses, source_metadata, target_bpm, target_fps, output_filename, debug)

        report = (f"Synced {source_metadata.get('title','')} → "
                  f"{target_frames} frames @ {target_bpm:.1f} BPM, {target_fps:.1f} FPS")
        return synced_poses, pose_video, report

    # ---------- Helpers ----------
    def _resample_and_sync(self, pose, meta, tgt_bpm, tgt_fps, method, loop, smooth, tgt_frames, debug):
        src_frames, pts, coords = pose.shape
        src_bpm   = meta.get("source_bpm", 120)
        src_fps   = meta.get("source_fps", 24)

        # ratios
        bpm_ratio = tgt_bpm / src_bpm
        fps_ratio = tgt_fps / src_fps

        if method == "frame_perfect":
            factor = (tgt_bpm / src_bpm) * (src_fps / tgt_fps)
        elif method == "beat_aligned":
            factor = (tgt_bpm / src_bpm)
        else:  # time_domain
            factor = tgt_bpm / src_bpm

        new_len = int(src_frames * factor)

        # handle loop/crop
        if loop == "loop_to_fit":
            loops = int(np.ceil(tgt_frames / new_len))
            pose = np.tile(pose, (loops, 1, 1))[:tgt_frames]
        elif loop == "crop_to_fit":
            pose = pose[:min(new_len, tgt_frames)]
            tgt_frames = pose.shape[0]

        # resample
        old_idx = np.linspace(0, pose.shape[0]-1, pose.shape[0])
        new_idx = np.linspace(0, pose.shape[0]-1, tgt_frames)
        synced = np.zeros((tgt_frames, pts, coords), dtype=np.float32)
        for i in range(pts):
            for c in range(coords):
                synced[:, i, c] = np.interp(new_idx, old_idx, pose[:, i, c])

        # smooth
        if smooth > 0:
            for t in range(1, len(synced)):
                synced[t] = smooth * synced[t-1] + (1 - smooth) * synced[t]

        return torch.from_numpy(synced).float()

    def _generate_pose_video(self, poses, w, h, bg, style, fps, debug):
        frames = []
        bg_color = {"black": (0, 0, 0), "white": (255, 255, 255), "transparent": (0, 0, 0)}[bg]
        for p in poses:
            img = np.full((h, w, 3), bg_color, dtype=np.uint8)
            kps = []
            for x, y in p[:23]:
                kps.append((int(np.clip(x*w,0,w-1)), int(np.clip(y*h,0,h-1))))
            if style in ("stickman", "skeleton"):
                for a, b in [(0,1),(1,3),(2,4),(5,6),(5,11),(6,12),(11,12),(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]:
                    if a < len(kps) and b < len(kps):
                        cv2.line(img, kps[a], kps[b], (255,255,255), 2)
                for pt in kps:
                    cv2.circle(img, pt, 4, (100,200,255), -1)
            elif style == "dots":
                for pt in kps:
                    cv2.circle(img, pt, 6, (0,255,0), -1)
            frames.append(torch.from_numpy(img.astype(np.float32)/255.0))
        return torch.stack(frames)

    def _save_synced_json(self, poses, meta, bpm, fps, filename, debug):
        lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dance_library")
        os.makedirs(lib, exist_ok=True)
        safe = "".join(c for c in filename if c.isalnum() or c in "-_").strip()
        path = os.path.join(lib, f"{safe}_synced.json")
        data = {
            "title": filename,
            "metadata": {"bpm": bpm, "fps": fps, "frame_count": len(poses), "sync_method": "auto"},
            "pose_tensor": poses.cpu().numpy().tolist()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        if debug:
            print("[BAIS1C] Saved synced →", path)

# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_MusicControlNet": BAIS1C_MusicControlNet}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_MusicControlNet": "🎵 BAIS1C Music Control Net (Auto-Sync)"}