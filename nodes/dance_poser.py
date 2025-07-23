import torch
import numpy as np
import librosa
import json
import os
import glob
from typing import List, Dict, Tuple, Optional

class BAIS1C_Suite_DancePoser:
    """
    BAIS1C Suite Dance Poser - Music Control Net (BASICS + AI)
    Part of BAIS1Cs_VACE_DANCE_SYNC_SUITE

    Takes pose library sequences and synchronizes them to music with 
    per-limb EQ frequency assignment and speed modifiers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # Pose Library Selection
                "library_dance": (cls._get_available_dances(), {"default": "none"}),
                # Music Sync Controls  
                "sync_mode": (["loop", "shuffle", "time_stretch"], {"default": "loop"}),
                "tempo_sync": ("BOOLEAN", {"default": True}),
                # EQ to Body Part Assignment
                "head_eq": (["beat", "bass", "mid", "high", "none"], {"default": "mid"}),
                "torso_eq": (["beat", "bass", "mid", "high", "none"], {"default": "bass"}), 
                "waist_eq": (["beat", "bass", "mid", "high", "none"], {"default": "bass"}),
                "legs_eq": (["beat", "bass", "mid", "high", "none"], {"default": "beat"}),
                "arms_eq": (["beat", "bass", "mid", "high", "none"], {"default": "high"}),
                # Speed Modifiers per Body Part
                "head_speed": (["0.5x", "1x", "1.5x"], {"default": "1x"}),
                "torso_speed": (["0.5x", "1x", "1.5x"], {"default": "1x"}),
                "waist_speed": (["0.5x", "1x", "1.5x"], {"default": "1x"}), 
                "legs_speed": (["0.5x", "1x", "1.5x"], {"default": "1x"}),
                "arms_speed": (["0.5x", "1x", "1.5x"], {"default": "1x"}),
                # Default/Idle Animation
                "default_idle": (["stand_arms_dangling", "bounce", "hands_on_hips"], {"default": "stand_arms_dangling"}),
                # Movement Controls
                "movement_smoothing": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "modulation_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tempo_factor": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.25}),
                # Visual Output
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 896, "min": 64, "max": 2048}),
                "fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                "background_color": (["black", "white", "gray"], {"default": "black"}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
                "show_joints": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_poses": ("POSE",),  # Direct pose tensor input (overrides library)
            }
        }

    RETURN_TYPES = ("IMAGE", "POSE")
    RETURN_NAMES = ("pose_video", "pose_tensor")
    FUNCTION = "sync_dance_to_music"
    CATEGORY = "BAIS1C Suite/Dance"

    @classmethod
    def _get_available_dances(cls) -> List[str]:
        """Scan dance_library folder for available JSON dance files"""
        try:
            suite_dir = os.path.dirname(__file__)
            library_dir = os.path.join(suite_dir, "../dance_library")
            if not os.path.exists(library_dir):
                return ["none", "starter_hiphop", "starter_ballet", "starter_freestyle"]
            json_files = glob.glob(os.path.join(library_dir, "*.json"))
            dance_names = ["none"]
            for json_file in json_files:
                basename = os.path.basename(json_file)
                dance_name = os.path.splitext(basename)[0]
                dance_names.append(dance_name)
            return dance_names if len(dance_names) > 1 else ["none", "starter_hiphop", "starter_ballet", "starter_freestyle"]
        except Exception as e:
            print(f"Warning: Could not scan dance library: {e}")
            return ["none", "starter_hiphop", "starter_ballet", "starter_freestyle"]

    def __init__(self):
        # Default idle animations (91-point pose format)
        self.idle_animations = {
            "stand_arms_dangling": self._create_stand_arms_dangling(),
            "bounce": self._create_bounce_idle(),
            "hands_on_hips": self._create_hands_on_hips()
        }
        self.speed_multipliers = {"0.5x": 0.5, "1x": 1.0, "1.5x": 1.5}
        self.pose_connections = [
            # Head/neck
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Feet
            (15, 17), (15, 18), (15, 19),
            (16, 20), (16, 21), (16, 22)
        ]

    def sync_dance_to_music(self, audio, library_dance, sync_mode, tempo_sync,
                           head_eq, torso_eq, waist_eq, legs_eq, arms_eq,
                           head_speed, torso_speed, waist_speed, legs_speed, arms_speed,
                           default_idle, movement_smoothing, modulation_intensity, tempo_factor,
                           width, height, fps, background_color, show_skeleton, show_joints, debug,
                           input_poses=None):
        if debug:
            print(f"[BAIS1C_Suite_DancePoser] Starting music sync...")
            print(f"Library dance: {library_dance}")
            print(f"Sync mode: {sync_mode}")

        audio_features = self._analyze_audio(audio, fps, debug)
        total_frames = audio_features["total_frames"]
        if debug:
            print(f"Audio analysis: {total_frames} frames, {audio_features['duration']:.2f}s, {audio_features['bpm']:.1f} BPM")

        if input_poses is not None:
            pose_sequence = input_poses.cpu().numpy() if isinstance(input_poses, torch.Tensor) else np.array(input_poses)
            if debug:
                print(f"Using direct pose input: {pose_sequence.shape}")
        else:
            pose_sequence = self._load_pose_from_library(library_dance, debug)

        fitted_poses = self._fit_poses_to_audio(pose_sequence, total_frames, sync_mode, debug)
        eq_assignments = {
            "head": head_eq, "torso": torso_eq, "waist": waist_eq, 
            "legs": legs_eq, "arms": arms_eq
        }
        speed_assignments = {
            "head": self.speed_multipliers[head_speed],
            "torso": self.speed_multipliers[torso_speed], 
            "waist": self.speed_multipliers[waist_speed],
            "legs": self.speed_multipliers[legs_speed],
            "arms": self.speed_multipliers[arms_speed]
        }

        modulated_poses = self._apply_music_modulation(
            fitted_poses, audio_features, eq_assignments, speed_assignments,
            movement_smoothing, modulation_intensity, tempo_factor, debug
        )

        frames = self._render_pose_sequence(
            modulated_poses, width, height, background_color, 
            show_skeleton, show_joints, debug
        )

        pose_tensor = torch.FloatTensor(modulated_poses)
        if debug:
            print(f"Final output: {len(frames)} frames, pose tensor {pose_tensor.shape}")

        return (frames, pose_tensor)

    # -- The rest of the code is unchanged, only method bodies below here --

    def _analyze_audio(self, audio, fps, debug=False):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if waveform.ndim == 3 and waveform.shape[0] >= 1:
            waveform = waveform[0]
        if waveform.ndim == 2:
            if waveform.shape[0] == 2:
                waveform = waveform.mean(axis=0)
            elif waveform.shape[0] == 1:
                waveform = waveform[0]
        duration = len(waveform) / sample_rate
        total_frames = int(duration * fps)
        try:
            tempo = librosa.beat.tempo(y=waveform, sr=sample_rate)
            bpm = float(tempo[0]) if hasattr(tempo, '__getitem__') else float(tempo)
            bpm = np.clip(bpm, 60, 200)
            _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate, units='frames', hop_length=512)
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=512)
            hop_length = int(sample_rate / fps)
            S = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
            bass_mask = (freqs >= 20) & (freqs < 250)
            mid_mask = (freqs >= 250) & (freqs < 2000)
            high_mask = (freqs >= 2000) & (freqs < 8000)
            def safe_normalize(arr):
                max_val = arr.max()
                return arr / max_val if max_val > 0 else np.zeros_like(arr)
            bass_energy = safe_normalize(S[bass_mask, :].mean(axis=0))
            mid_energy = safe_normalize(S[mid_mask, :].mean(axis=0))
            high_energy = safe_normalize(S[high_mask, :].mean(axis=0))
        except Exception as e:
            if debug:
                print(f"Audio analysis failed: {e}, using fallback")
            amplitude = np.abs(waveform) / (np.max(np.abs(waveform)) + 1e-8)
            beat_times = []
            bpm = 120.0
            bass_energy = amplitude[:total_frames] if len(amplitude) >= total_frames else np.pad(amplitude, (0, max(0, total_frames - len(amplitude))))
            mid_energy = bass_energy.copy()
            high_energy = bass_energy * 0.5
        beat_strength = np.zeros(total_frames)
        for bt in beat_times:
            if bt <= duration:
                idx = int((bt / duration) * total_frames)
                if 0 <= idx < total_frames:
                    beat_strength[idx] = 1.0
        if len(bass_energy) != total_frames:
            frame_indices = np.linspace(0, len(bass_energy) - 1, total_frames).astype(np.int32)
            frame_indices = np.clip(frame_indices, 0, len(bass_energy) - 1)
            bass_energy = bass_energy[frame_indices]
            mid_energy = mid_energy[frame_indices]
            high_energy = high_energy[frame_indices]
        return {
            "beat_strength": beat_strength,
            "bass": bass_energy,
            "mid": mid_energy,
            "high": high_energy,
            "total_frames": total_frames,
            "duration": duration,
            "bpm": bpm
        }

    def _load_pose_from_library(self, library_dance, debug=False):
        if library_dance == "none" or library_dance.startswith("starter_"):
            if debug:
                print(f"Using built-in animation: {library_dance}")
            return self._create_starter_animation(library_dance)
        try:
            suite_dir = os.path.dirname(__file__)
            library_dir = os.path.join(suite_dir, "../dance_library")
            json_path = os.path.join(library_dir, f"{library_dance}.json")
            if not os.path.exists(json_path):
                if debug:
                    print(f"JSON file not found: {json_path}, using default")
                return self._create_starter_animation("starter_hiphop")
            with open(json_path, 'r') as f:
                data = json.load(f)
            if "pose_tensor" in data:
                poses = np.array(data["pose_tensor"])
                if debug:
                    print(f"Loaded pose sequence: {poses.shape}")
                return poses
            else:
                if debug:
                    print("No pose_tensor in JSON, using default")
                return self._create_starter_animation("starter_hiphop")
        except Exception as e:
            if debug:
                print(f"Failed to load {library_dance}: {e}")
            return self._create_starter_animation("starter_hiphop")

    def _create_starter_animation(self, animation_name):
        if animation_name == "starter_hiphop":
            return self._create_hiphop_sequence()
        elif animation_name == "starter_ballet":
            return self._create_ballet_sequence()
        elif animation_name == "starter_freestyle":
            return self._create_freestyle_sequence()
        else:
            return self._create_default_sequence()

    def _create_default_sequence(self):
        base_pose = self._create_base_91_pose()
        sequence = []
        for i in range(4):
            pose = base_pose.copy()
            pose[7][1] += 0.1 * np.sin(i * np.pi / 2)
            pose[8][1] += 0.1 * np.cos(i * np.pi / 2)
            sequence.append(pose)
        return np.array(sequence)

    def _create_hiphop_sequence(self):
        base_pose = self._create_base_91_pose()
        sequence = []
        for i in range(8):
            pose = base_pose.copy()
            t = i / 8.0
            pose[7][0] += 0.15 * np.sin(t * 4 * np.pi)
            pose[8][0] -= 0.15 * np.sin(t * 4 * np.pi)
            pose[9][1] += 0.2 * np.abs(np.sin(t * 2 * np.pi))
            pose[10][1] += 0.2 * np.abs(np.cos(t * 2 * np.pi))
            pose[11][0] += 0.08 * np.sin(t * 2 * np.pi)
            pose[12][0] -= 0.08 * np.sin(t * 2 * np.pi)
            sequence.append(pose)
        return np.array(sequence)

    def _create_ballet_sequence(self):
        base_pose = self._create_base_91_pose()
        sequence = []
        for i in range(6):
            pose = base_pose.copy()
            t = i / 6.0
            pose[7][0] -= 0.1
            pose[8][0] += 0.1
            pose[7][1] -= 0.05
            pose[8][1] -= 0.05
            pose[9][0] -= 0.15
            pose[10][0] += 0.15
            pose[9][1] -= 0.1 + 0.05 * np.sin(t * 2 * np.pi)
            pose[10][1] -= 0.1 + 0.05 * np.cos(t * 2 * np.pi)
            pose[15][0] -= 0.02
            pose[16][0] += 0.02
            sequence.append(pose)
        return np.array(sequence)

    def _create_freestyle_sequence(self):
        base_pose = self._create_base_91_pose()
        sequence = []
        for i in range(12):
            pose = base_pose.copy()
            t = i / 12.0
            pose[7][0] += 0.2 * np.sin(t * 6 * np.pi + 0.5)
            pose[8][0] += 0.2 * np.cos(t * 5 * np.pi)
            pose[7][1] += 0.15 * np.cos(t * 3 * np.pi)
            pose[8][1] += 0.15 * np.sin(t * 4 * np.pi + 1)
            pose[9][0] += 0.1 * np.sin(t * 8 * np.pi)
            pose[10][0] += 0.1 * np.cos(t * 7 * np.pi)
            pose[0][0] += 0.05 * np.sin(t * 3 * np.pi)
            sequence.append(pose)
        return np.array(sequence)

    def _create_base_91_pose(self):
        body_pose = np.array([
            [0.5, 0.12],   # 0: nose
            [0.5, 0.20],   # 1: neck  
            [0.53, 0.20],  # 2: right eye
            [0.48, 0.23],  # 3: left ear
            [0.56, 0.23],  # 4: right ear
            [0.40, 0.28],  # 5: left shoulder
            [0.60, 0.28],  # 6: right shoulder
            [0.32, 0.48],  # 7: left elbow
            [0.68, 0.48],  # 8: right elbow
            [0.23, 0.68],  # 9: left wrist
            [0.77, 0.68],  # 10: right wrist
            [0.44, 0.62],  # 11: left hip
            [0.56, 0.62],  # 12: right hip
            [0.39, 0.84],  # 13: left knee
            [0.61, 0.84],  # 14: right knee
            [0.37, 1.00],  # 15: left ankle
            [0.63, 1.00],  # 16: right ankle
            [0.35, 1.05], [0.39, 1.07], [0.36, 1.02],  # 17-19: left foot
            [0.65, 1.05], [0.61, 1.07], [0.64, 1.02],  # 20-22: right foot
        ])
        face_center = np.array([0.5, 0.13])
        face_radius = 0.045
        face_points = []
        for i in range(68):
            angle = 2 * np.pi * i / 68
            x = face_center[0] + np.cos(angle) * face_radius * (0.8 + 0.4 * (i % 3) / 3)
            y = face_center[1] + np.sin(angle) * face_radius * (0.6 + 0.4 * (i % 2))
            face_points.append([x, y])
        full_pose = np.concatenate([body_pose, np.array(face_points)], axis=0)
        return np.clip(full_pose, 0.0, 1.0)

    def _fit_poses_to_audio(self, pose_sequence, total_frames, sync_mode, debug=False):
        if len(pose_sequence) == 0:
            pose_sequence = self._create_default_sequence()
        pose_frames = len(pose_sequence)
        if debug:
            print(f"Fitting {pose_frames} pose frames to {total_frames} audio frames, mode: {sync_mode}")
        if sync_mode == "loop":
            if pose_frames >= total_frames:
                return pose_sequence[:total_frames]
            else:
                loops_needed = (total_frames + pose_frames - 1) // pose_frames
                looped = np.tile(pose_sequence, (loops_needed, 1, 1))
                return looped[:total_frames]
        elif sync_mode == "shuffle":
            fitted = []
            while len(fitted) < total_frames:
                shuffled_poses = pose_sequence.copy()
                np.random.shuffle(shuffled_poses)
                fitted.extend(shuffled_poses)
            return np.array(fitted[:total_frames])
        elif sync_mode == "time_stretch":
            if pose_frames == total_frames:
                return pose_sequence
            old_indices = np.linspace(0, pose_frames - 1, pose_frames)
            new_indices = np.linspace(0, pose_frames - 1, total_frames)
            stretched = np.zeros((total_frames, pose_sequence.shape[1], pose_sequence.shape[2]))
            for point_idx in range(pose_sequence.shape[1]):
                for coord_idx in range(pose_sequence.shape[2]):
                    stretched[:, point_idx, coord_idx] = np.interp(
                        new_indices, old_indices, pose_sequence[:, point_idx, coord_idx]
                    )
            return stretched
        else:
            return pose_sequence[:total_frames] if len(pose_sequence) >= total_frames else np.tile(pose_sequence, (2, 1, 1))[:total_frames]

    def _apply_music_modulation(self, poses, audio_features, eq_assignments, speed_assignments,
                               smoothing, intensity, tempo_factor, debug=False):
        modulated_poses = poses.copy()
        total_frames = len(poses)
        if debug:
            print(f"Applying music modulation to {total_frames} frames")
            print(f"EQ assignments: {eq_assignments}")
            print(f"Speed assignments: {speed_assignments}")
        beat = audio_features["beat_strength"]
        bass = audio_features["bass"]
        mid = audio_features["mid"]
        high = audio_features["high"]
        signal_map = {"beat": beat, "bass": bass, "mid": mid, "high": high, "none": np.zeros(total_frames)}
        for frame_i in range(total_frames):
            frame_signals = {}
            for limb, eq_source in eq_assignments.items():
                signal = signal_map[eq_source][frame_i] if frame_i < len(signal_map[eq_source]) else 0.0
                speed_mult = speed_assignments[limb]
                frame_signals[limb] = signal * intensity * speed_mult
            pose = modulated_poses[frame_i]
            # HEAD (0-4): Nod, sway, bob
            if frame_signals["head"] > 0.1:
                head_mod = frame_signals["head"] * 0.02
                for head_idx in range(5):
                    if head_idx < len(pose):
                        pose[head_idx][1] += head_mod * np.sin(frame_i * 0.3)
                        pose[head_idx][0] += head_mod * 0.5 * np.cos(frame_i * 0.2)
            # TORSO (5-6, 11-12): Shoulders and hips sway
            if frame_signals["torso"] > 0.1:
                torso_mod = frame_signals["torso"] * 0.03
                sway = torso_mod * np.sin(frame_i * 0.15)
                if 5 < len(pose): pose[5][0] += sway
                if 6 < len(pose): pose[6][0] -= sway
                if 11 < len(pose): pose[11][0] += sway * 0.7
                if 12 < len(pose): pose[12][0] -= sway * 0.7
            # WAIST: Hip center movement
            if frame_signals["waist"] > 0.1:
                waist_mod = frame_signals["waist"] * 0.025
                hip_movement = waist_mod * np.sin(frame_i * 0.12)
                if 11 < len(pose): pose[11][0] += hip_movement
                if 12 < len(pose): pose[12][0] += hip_movement
            # LEGS (13-22): Steps, bounces, knee bends
            if frame_signals["legs"] > 0.1:
                leg_mod = frame_signals["legs"] * 0.04
                step_phase = 1 if (frame_i // 4) % 2 == 0 else -1
                if 13 < len(pose): pose[13][1] += leg_mod * step_phase * 0.5
                if 14 < len(pose): pose[14][1] -= leg_mod * step_phase * 0.5
                if 15 < len(pose): pose[15][1] += leg_mod * step_phase * 0.3
                if 16 < len(pose): pose[16][1] -= leg_mod * step_phase * 0.3
                for foot_idx in range(17, 23):
                    if foot_idx < len(pose):
                        ankle_ref = 15 if foot_idx < 20 else 16
                        if ankle_ref < len(pose):
                            pose[foot_idx][1] = pose[ankle_ref][1] + 0.05
            # ARMS (7-10): Waves, pumps, reaches
            if frame_signals["arms"] > 0.1:
                arm_mod = frame_signals["arms"] * 0.05
                if 7 < len(pose): pose[7][1] += arm_mod * np.sin(frame_i * 0.25)
                if 8 < len(pose): pose[8][1] += arm_mod * np.cos(frame_i * 0.25)
                if 7 < len(pose): pose[7][0] += arm_mod * 0.5 * np.cos(frame_i * 0.2)
                if 8 < len(pose): pose[8][0] -= arm_mod * 0.5 * np.cos(frame_i * 0.2)
                if 9 < len(pose): pose[9][1] += arm_mod * np.cos(frame_i * 0.3 + 0.5)
                if 10 < len(pose): pose[10][1] += arm_mod * np.sin(frame_i * 0.3 + 0.5)
            modulated_poses[frame_i] = np.clip(pose, 0.0, 1.0)
        if smoothing > 0:
            for frame_i in range(1, total_frames):
                modulated_poses[frame_i] = (smoothing * modulated_poses[frame_i] + 
                                          (1 - smoothing) * modulated_poses[frame_i - 1])
        return modulated_poses

    def _render_pose_sequence(self, pose_sequence, width, height, background_color,
                            show_skeleton, show_joints, debug=False):
        bg_colors = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}
        bg_color = bg_colors.get(background_color, (0, 0, 0))
        skel_color = (255, 255, 255) if background_color != "white" else (0, 0, 0)
        joint_color = (140, 200, 255) if background_color != "white" else (80, 80, 80)
        frames = []
        for frame_idx, pose in enumerate(pose_sequence):
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            keypoints = []
            for x, y in pose:
                px = int(np.clip(x * width, 0, width - 1))
                py = int(np.clip(y * height, 0, height - 1))
                keypoints.append((px, py))
            if show_skeleton and len(keypoints) >= 23:
                for conn in self.pose_connections:
                    i0, i1 = conn
                    if i0 < len(keypoints) and i1 < len(keypoints):
                        try:
                            import cv2
                            cv2.line(frame, keypoints[i0], keypoints[i1], skel_color, 2)
                        except:
                            pass
            if show_joints:
                for i in range(min(23, len(keypoints))):
                    try:
                        import cv2
                        cv2.circle(frame, keypoints[i], 4, joint_color, -1)
                    except:
                        pass
                for i in range(23, min(91, len(keypoints))):
                    try:
                        import cv2
                        cv2.circle(frame, keypoints[i], 1, (255, 180, 180), -1)
                    except:
                        pass
            if debug and frame_idx < 5:
                import cv2
                cv2.putText(frame, f"Frame {frame_idx} - {len(keypoints)} pts", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
        return torch.stack(frames)

    def _create_stand_arms_dangling(self):
        return self._create_base_91_pose()

    def _create_bounce_idle(self):
        base_pose = self._create_base_91_pose()
        sequence = []
        for i in range(6):
            pose = base_pose.copy()
            bounce = 0.02 * np.sin(i * np.pi / 3)
            for point_idx in range(23):
                pose[point_idx][1] += bounce
            sequence.append(pose)
        return np.array(sequence)

    def _create_hands_on_hips(self):
        base_pose = self._create_base_91_pose()
        base_pose[7] = [0.35, 0.40]
        base_pose[8] = [0.65, 0.40]
        base_pose[9] = [0.38, 0.62]
        base_pose[10] = [0.62, 0.62]
        return base_pose

# --- Node registration at end ---
NODE_CLASS_MAPPINGS = {"BAIS1C_Suite_DancePoser": BAIS1C_Suite_DancePoser}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_Suite_DancePoser": "🕺 BAIS1C Suite Dance Poser - Music Control Net"}
