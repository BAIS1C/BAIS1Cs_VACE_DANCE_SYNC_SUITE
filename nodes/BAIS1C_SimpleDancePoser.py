import torch
import numpy as np
import librosa
import json
import os
import glob
import cv2
from typing import List, Dict, Tuple, Optional

class BAIS1C_SimpleDancePoser:
    """
    BAIS1C VACE Dance Sync Suite - Simple Dance Poser
    
    Creative experimentation node for generating animated dance sequences.
    Allows users to play with speed, smoothing, and basic music reactivity
    to create their own custom dance animations from library poses or built-in moves.
    
    This is the "play and experiment" version - not for precise sync, but for creativity!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                
                # Dance Source
                "dance_source": (["library", "built_in"], {"default": "built_in"}),
                "library_dance": (cls._get_available_dances(), {"default": "none"}),
                "built_in_style": (["hiphop", "ballet", "freestyle", "bounce", "robot"], {"default": "hiphop"}),
                
                # Creative Controls
                "dance_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "movement_smoothing": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05}),
                "music_reactivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "animation_loops": ("INT", {"default": 4, "min": 1, "max": 20}),
                
                # Simple Music Response
                "react_to": (["beat", "bass", "energy", "none"], {"default": "beat"}),
                "reaction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # Visual Output
                "output_fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "height": ("INT", {"default": 896, "min": 256, "max": 1024}),
                "visualization": (["stickman", "dots", "skeleton", "none"], {"default": "stickman"}),
                "background": (["black", "white", "dark_blue"], {"default": "black"}),
                
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_pose_tensor": ("POSE",),  # Optional direct input
            }
        }

    RETURN_TYPES = ("POSE", "IMAGE", "STRING")
    RETURN_NAMES = ("animated_poses", "dance_video", "creation_info")
    FUNCTION = "create_dance_animation"
    CATEGORY = "BAIS1C VACE Suite/Creative"

    @classmethod
    def _get_available_dances(cls) -> List[str]:
        """Get available dances from library"""
        try:
            current_file = os.path.abspath(__file__)
            suite_dir = None
            
            check_dir = os.path.dirname(current_file)
            for _ in range(5):
                if os.path.exists(os.path.join(check_dir, 'dance_library')):
                    suite_dir = check_dir
                    break
                parent_dir = os.path.dirname(check_dir)
                if parent_dir == check_dir:
                    break
                check_dir = parent_dir
            
            if not suite_dir:
                return ["none"]
            
            library_dir = os.path.join(suite_dir, "dance_library")
            json_files = glob.glob(os.path.join(library_dir, "*.json"))
            
            dance_names = ["none"]
            for json_file in json_files:
                basename = os.path.basename(json_file)
                dance_name = os.path.splitext(basename)[0]
                dance_names.append(dance_name)
                
            return dance_names if len(dance_names) > 1 else ["none"]
            
        except Exception:
            return ["none"]

    def __init__(self):
        """Initialize with built-in dance sequences"""
        # Create built-in dance animations
        self.built_in_dances = {
            "hiphop": self._create_hiphop_sequence(),
            "ballet": self._create_ballet_sequence(), 
            "freestyle": self._create_freestyle_sequence(),
            "bounce": self._create_bounce_sequence(),
            "robot": self._create_robot_sequence()
        }
        
        # Pose connections for visualization
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

    def create_dance_animation(self, audio, dance_source, library_dance, built_in_style,
                             dance_speed, movement_smoothing, music_reactivity, animation_loops,
                             react_to, reaction_strength, output_fps, width, height, 
                             visualization, background, debug, input_pose_tensor=None):
        """
        Create animated dance sequence with user controls
        """
        if debug:
            print(f"\n[BAIS1C Simple Dance Poser] === CREATING DANCE ===")
            print(f"Style: {built_in_style if dance_source == 'built_in' else library_dance}")
            print(f"Speed: {dance_speed}x, Smoothing: {movement_smoothing}, Reactivity: {music_reactivity}")

        # Step 1: Get base pose sequence
        base_poses = self._get_base_poses(dance_source, library_dance, built_in_style, input_pose_tensor, debug)
        
        if base_poses is None:
            error_msg = "Failed to load dance poses"
            empty_tensor = torch.zeros((1, 128, 2), dtype=torch.float32)
            empty_video = torch.zeros((1, height, width, 3), dtype=torch.float32) 
            return (empty_tensor, empty_video, error_msg)

        # Step 2: Analyze audio for basic reactivity
        audio_features = self._analyze_audio_simple(audio, output_fps, debug)
        
        # Step 3: Create animation sequence with user parameters
        animated_poses = self._create_animation_sequence(
            base_poses, audio_features, dance_speed, animation_loops, 
            music_reactivity, react_to, reaction_strength, debug
        )
        
        # Step 4: Apply smoothing
        if movement_smoothing > 0:
            animated_poses = self._apply_smoothing(animated_poses, movement_smoothing, debug)
        
        # Step 5: Generate visualization
        dance_video = None
        if visualization != "none":
            dance_video = self._create_dance_video(
                animated_poses, width, height, background, visualization, output_fps, debug
            )
        else:
            dance_video = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        # Step 6: Create info report
        creation_info = self._create_info_report(
            base_poses, animated_poses, dance_speed, movement_smoothing, 
            music_reactivity, audio_features
        )
        
        if debug:
            print(f"[BAIS1C Simple Dance Poser] === DANCE CREATED ===")
            print(f"Output: {animated_poses.shape[0]} frames @ {output_fps} FPS")
        
        return (animated_poses, dance_video, creation_info)

    def _get_base_poses(self, dance_source, library_dance, built_in_style, input_pose_tensor, debug):
        """Get base pose sequence from various sources"""
        
        if input_pose_tensor is not None:
            if debug:
                print(f"[BAIS1C Simple Dance Poser] Using direct tensor input: {input_pose_tensor.shape}")
            
            if isinstance(input_pose_tensor, torch.Tensor):
                return input_pose_tensor.cpu().numpy()
            else:
                return np.array(input_pose_tensor)
        
        elif dance_source == "built_in":
            if debug:
                print(f"[BAIS1C Simple Dance Poser] Using built-in style: {built_in_style}")
            return self.built_in_dances.get(built_in_style, self.built_in_dances["hiphop"])
        
        elif dance_source == "library" and library_dance != "none":
            return self._load_from_library(library_dance, debug)
        
        else:
            if debug:
                print(f"[BAIS1C Simple Dance Poser] No valid source, using default hiphop")
            return self.built_in_dances["hiphop"]

    def _load_from_library(self, library_dance, debug):
        """Load pose sequence from JSON library"""
        try:
            current_file = os.path.abspath(__file__)
            suite_dir = None
            
            check_dir = os.path.dirname(current_file)
            for _ in range(5):
                if os.path.exists(os.path.join(check_dir, 'dance_library')):
                    suite_dir = check_dir
                    break
                parent_dir = os.path.dirname(check_dir)
                if parent_dir == check_dir:
                    break
                check_dir = parent_dir
            
            if not suite_dir:
                return None
            
            json_path = os.path.join(suite_dir, "dance_library", f"{library_dance}.json")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if "pose_tensor" in data:
                poses = np.array(data["pose_tensor"], dtype=np.float32)
                if debug:
                    print(f"[BAIS1C Simple Dance Poser] Loaded from library: {poses.shape}")
                return poses
            else:
                return None
                
        except Exception as e:
            if debug:
                print(f"[BAIS1C Simple Dance Poser] Failed to load {library_dance}: {e}")
            return None

    def _analyze_audio_simple(self, audio, fps, debug):
        """Simple audio analysis for basic reactivity"""
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        
        duration = len(waveform) / sample_rate
        total_frames = int(duration * fps)
        
        try:
            # Basic beat detection
            _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
            
            # Create beat strength array
            beat_strength = np.zeros(total_frames)
            for bt in beat_times:
                if bt <= duration:
                    idx = int((bt / duration) * total_frames)
                    if 0 <= idx < total_frames:
                        beat_strength[idx] = 1.0
            
            # Simple frequency analysis
            hop_length = int(sample_rate / fps)
            S = np.abs(librosa.stft(waveform, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=sample_rate)
            
            # Basic frequency bands
            bass_mask = freqs < 250
            bass_energy = S[bass_mask, :].mean(axis=0)
            
            # Overall energy
            energy = np.mean(S, axis=0)
            
            # Normalize
            bass_energy = bass_energy / (np.max(bass_energy) + 1e-8)
            energy = energy / (np.max(energy) + 1e-8)
            
            # Resize to match frame count
            if len(bass_energy) != total_frames:
                frame_indices = np.linspace(0, len(bass_energy) - 1, total_frames).astype(np.int32)
                frame_indices = np.clip(frame_indices, 0, len(bass_energy) - 1)
                bass_energy = bass_energy[frame_indices]
                energy = energy[frame_indices]
            
        except Exception as e:
            if debug:
                print(f"[BAIS1C Simple Dance Poser] Audio analysis failed: {e}, using defaults")
            beat_strength = np.random.rand(total_frames) * 0.5
            bass_energy = np.random.rand(total_frames) * 0.5
            energy = np.random.rand(total_frames) * 0.5
        
        return {
            "beat": beat_strength,
            "bass": bass_energy,
            "energy": energy,
            "total_frames": total_frames,
            "duration": duration
        }

    def _create_animation_sequence(self, base_poses, audio_features, dance_speed, animation_loops,
                                 music_reactivity, react_to, reaction_strength, debug):
        """Create animation sequence with user parameters"""
        
        target_frames = audio_features["total_frames"]
        base_frames = len(base_poses)
        
        # Apply speed modification
        speed_adjusted_frames = int(base_frames / dance_speed)
        if speed_adjusted_frames < 1:
            speed_adjusted_frames = 1
        
        # Create speed-adjusted sequence
        if dance_speed != 1.0:
            speed_poses = self._resample_poses(base_poses, speed_adjusted_frames)
        else:
            speed_poses = base_poses
        
        # Loop to fill target duration
        loops_needed = max(1, int(np.ceil(target_frames / len(speed_poses))))
        looped_poses = np.tile(speed_poses, (loops_needed, 1, 1))
        
        # Crop to target length
        if len(looped_poses) > target_frames:
            looped_poses = looped_poses[:target_frames]
        elif len(looped_poses) < target_frames:
            # Pad with last frame
            padding = np.tile(looped_poses[-1:], (target_frames - len(looped_poses), 1, 1))
            looped_poses = np.concatenate([looped_poses, padding], axis=0)
        
        # Apply music reactivity
        if music_reactivity > 0 and react_to != "none":
            looped_poses = self._apply_music_reactivity(
                looped_poses, audio_features, react_to, reaction_strength, music_reactivity, debug
            )
        
        if debug:
            print(f"[BAIS1C Simple Dance Poser] Animation sequence:")
            print(f"  Base: {base_frames} → Speed adjusted: {speed_adjusted_frames}")
            print(f"  Looped: {len(looped_poses)} frames")
            print(f"  Music reactivity: {music_reactivity} ({react_to})")
        
        return torch.from_numpy(looped_poses).float()

    def _apply_music_reactivity(self, poses, audio_features, react_to, reaction_strength, 
                              music_reactivity, debug):
        """Apply simple music reactivity to poses"""
        
        signal = audio_features.get(react_to, audio_features["energy"])
        modulated_poses = poses.copy()
        
        for frame_idx, pose in enumerate(poses):
            if frame_idx >= len(signal):
                continue
                
            # Get signal strength for this frame
            signal_strength = signal[frame_idx] * reaction_strength * music_reactivity
            
            if signal_strength > 0.1:
                # Apply simple movements based on signal
                movement = signal_strength * 0.02  # Small movement
                
                # Head nod
                if len(pose) > 0:
                    modulated_poses[frame_idx, 0, 1] += movement * np.sin(frame_idx * 0.5)
                
                # Shoulder bounce
                if len(pose) > 6:
                    modulated_poses[frame_idx, 5, 1] -= movement * 0.5
                    modulated_poses[frame_idx, 6, 1] -= movement * 0.5
                
                # Hip sway
                if len(pose) > 12:
                    sway = movement * np.sin(frame_idx * 0.3)
                    modulated_poses[frame_idx, 11, 0] += sway
                    modulated_poses[frame_idx, 12, 0] -= sway
                
                # Arm movement
                if len(pose) > 10:
                    arm_move = movement * np.cos(frame_idx * 0.4)
                    modulated_poses[frame_idx, 9, 1] += arm_move
                    modulated_poses[frame_idx, 10, 1] += arm_move
        
        # Keep poses in valid range
        modulated_poses = np.clip(modulated_poses, 0.0, 1.0)
        
        return modulated_poses

    def _apply_smoothing(self, poses, smoothing_factor, debug):
        """Apply temporal smoothing to reduce jitter"""
        if isinstance(poses, torch.Tensor):
            poses_np = poses.cpu().numpy()
        else:
            poses_np = poses
        
        smoothed = poses_np.copy()
        
        # Exponential moving average smoothing
        for frame_idx in range(1, len(poses_np)):
            smoothed[frame_idx] = (
                smoothing_factor * smoothed[frame_idx - 1] +
                (1 - smoothing_factor) * poses_np[frame_idx]
            )
        
        if debug:
            print(f"[BAIS1C Simple Dance Poser] Applied smoothing: {smoothing_factor}")
        
        return torch.from_numpy(smoothed).float()

    def _resample_poses(self, poses, target_frames):
        """Resample pose sequence to target frame count"""
        if len(poses) == target_frames:
            return poses
        
        original_frames = len(poses)
        old_indices = np.linspace(0, original_frames - 1, original_frames)
        new_indices = np.linspace(0, original_frames - 1, target_frames)
        
        resampled = np.zeros((target_frames, poses.shape[1], poses.shape[2]), dtype=np.float32)
        
        for point_idx in range(poses.shape[1]):
            for coord_idx in range(poses.shape[2]):
                resampled[:, point_idx, coord_idx] = np.interp(
                    new_indices, old_indices, poses[:, point_idx, coord_idx]
                )
        
        return resampled

    def _create_dance_video(self, poses, width, height, background, style, fps, debug):
        """Create dance visualization video"""
        if isinstance(poses, torch.Tensor):
            poses_np = poses.cpu().numpy()
        else:
            poses_np = poses
        
        bg_colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255), 
            "dark_blue": (20, 20, 40)
        }
        bg_color = bg_colors.get(background, (0, 0, 0))
        
        frames = []
        
        for frame_idx, pose in enumerate(poses_np):
            # Create frame
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            
            # Convert normalized coordinates to pixels
            keypoints = []
            for x, y in pose[:23]:  # Use first 23 points (body)
                px = int(np.clip(x * width, 0, width - 1))
                py = int(np.clip(y * height, 0, height - 1))
                keypoints.append((px, py))
            
            # Draw based on style
            if style == "stickman" or style == "skeleton":
                # Draw skeleton connections
                for i, j in self.pose_connections:
                    if i < len(keypoints) and j < len(keypoints):
                        cv2.line(frame, keypoints[i], keypoints[j], (255, 255, 255), 2)
                
                # Draw joints
                for i, point in enumerate(keypoints):
                    color = (100, 200, 255) if i < 5 else (255, 200, 100)
                    cv2.circle(frame, point, 4, color, -1)
                    
            elif style == "dots":
                # Draw only dots
                for i, point in enumerate(keypoints):
                    if i < 5:  # Head
                        color = (255, 100, 100)
                        radius = 8
                    elif i < 12:  # Torso/arms
                        color = (100, 255, 100)
                        radius = 6
                    else:  # Legs
                        color = (100, 100, 255)
                        radius = 6
                    cv2.circle(frame, point, radius, color, -1)
            
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
        
        if debug:
            print(f"[BAIS1C Simple Dance Poser] Created {len(frames)} video frames ({style} style)")
        
        return torch.stack(frames)

    def _create_info_report(self, base_poses, animated_poses, dance_speed, movement_smoothing,
                          music_reactivity, audio_features):
        """Create info report about the generated dance"""
        
        base_frames = len(base_poses) if base_poses is not None else 0
        output_frames = len(animated_poses)
        duration = audio_features["duration"]
        
        report = f"""BAIS1C Simple Dance Poser - Creation Report
==========================================
Base Animation: {base_frames} frames
Output Animation: {output_frames} frames ({duration:.1f}s)

Creative Settings:
  Dance Speed: {dance_speed}x
  Movement Smoothing: {movement_smoothing:.2f}
  Music Reactivity: {music_reactivity:.2f}

Audio Analysis:
  Duration: {duration:.1f} seconds
  Total Frames: {audio_features['total_frames']}
  Beat Detection: {'✅ Active' if music_reactivity > 0 else '❌ Disabled'}

Result: Animated dance sequence ready for playback!
"""
        
        return report

    # Built-in dance creation methods
    def _create_base_pose(self):
        """Create base T-pose for dance sequences"""
        # Simplified 128-point pose (focusing on main body points)
        pose = np.zeros((128, 2), dtype=np.float32)
        
        # Basic body keypoints (first 23 points)
        body_pose = np.array([
            [0.5, 0.12],   # 0: nose
            [0.5, 0.20],   # 1: neck  
            [0.48, 0.12],  # 2: right eye
            [0.52, 0.12],  # 3: left eye
            [0.46, 0.14],  # 4: right ear
            [0.54, 0.14],  # 5: left ear
            [0.40, 0.28],  # 6: left shoulder
            [0.60, 0.28],  # 7: right shoulder
            [0.32, 0.48],  # 8: left elbow
            [0.68, 0.48],  # 9: right elbow
            [0.23, 0.68],  # 10: left wrist
            [0.77, 0.68],  # 11: right wrist
            [0.44, 0.62],  # 12: left hip
            [0.56, 0.62],  # 13: right hip
            [0.39, 0.84],  # 14: left knee
            [0.61, 0.84],  # 15: right knee
            [0.37, 1.00],  # 16: left ankle
            [0.63, 1.00],  # 17: right ankle
            [0.35, 1.05], [0.39, 1.07], [0.36, 1.02],  # 18-20: left foot
            [0.65, 1.05], [0.61, 1.07], [0.64, 1.02],  # 21-23: right foot
        ], dtype=np.float32)
        
        pose[:len(body_pose)] = body_pose
        
        # Fill remaining points with reasonable defaults
        # Face points (simplified)
        face_center = np.array([0.5, 0.13])
        for i in range(24, 92):  # Face keypoints
            angle = 2 * np.pi * (i - 24) / 68
            offset = 0.02
            pose[i] = [
                face_center[0] + offset * np.cos(angle),
                face_center[1] + offset * np.sin(angle) * 0.7
            ]
        
        # Hand points (simplified)
        left_wrist = pose[10]
        right_wrist = pose[11]
        
        for i in range(92, 113):  # Left hand
            offset_idx = i - 92
            pose[i] = [
                left_wrist[0] + (offset_idx % 5) * 0.005 - 0.01,
                left_wrist[1] + (offset_idx // 5) * 0.005
            ]
        
        for i in range(113, 128):  # Right hand  
            offset_idx = i - 113
            pose[i] = [
                right_wrist[0] + (offset_idx % 5) * 0.005 - 0.01,
                right_wrist[1] + (offset_idx // 5) * 0.005
            ]
        
        return np.clip(pose, 0.0, 1.0)

    def _create_hiphop_sequence(self):
        """Create hip-hop style dance sequence"""
        base_pose = self._create_base_pose()
        sequence = []
        
        for i in range(8):
            pose = base_pose.copy()
            t = i / 8.0
            
            # Hip-hop arm movements
            pose[8][0] += 0.15 * np.sin(t * 4 * np.pi)  # Left elbow
            pose[9][0] -= 0.15 * np.sin(t * 4 * np.pi)  # Right elbow
            pose[10][1] += 0.2 * np.abs(np.sin(t * 2 * np.pi))  # Left wrist
            pose[11][1] += 0.2 * np.abs(np.cos(t * 2 * np.pi))  # Right wrist
            
            # Hip movement
            pose[12][0] += 0.08 * np.sin(t * 2 * np.pi)  # Left hip
            pose[13][0] -= 0.08 * np.sin(t * 2 * np.pi)  # Right hip
            
            sequence.append(pose)
        
        return np.array(sequence)

    def _create_ballet_sequence(self):
        """Create ballet style dance sequence"""
        base_pose = self._create_base_pose()
        sequence = []
        
        for i in range(6):
            pose = base_pose.copy()
            t = i / 6.0
            
            # Graceful arm positions
            pose[8][0] -= 0.1   # Left elbow inward
            pose[9][0] += 0.1   # Right elbow inward
            pose[8][1] -= 0.05  # Elbows slightly up
            pose[9][1] -= 0.05
            
            # Graceful hand positions
            pose[10][0] -= 0.15  # Left hand
            pose[11][0] += 0.15  # Right hand
            pose[10][1] -= 0.1 + 0.05 * np.sin(t * 2 * np.pi)
            pose[11][1] -= 0.1 + 0.05 * np.cos(t * 2 * np.pi)
            
            # Slight foot positioning
            pose[16][0] -= 0.02  # Left ankle
            pose[17][0] += 0.02  # Right ankle
            
            sequence.append(pose)
        
        return np.array(sequence)

    def _create_freestyle_sequence(self):
        """Create freestyle dance sequence"""
        base_pose = self._create_base_pose()
        sequence = []
        
        for i in range(12):
            pose = base_pose.copy()
            t = i / 12.0
            
            # Free-flowing movements
            pose[8][0] += 0.2 * np.sin(t * 6 * np.pi + 0.5)   # Left elbow
            pose[9][0] += 0.2 * np.cos(t * 5 * np.pi)        # Right elbow
            pose[8][1] += 0.15 * np.cos(t * 3 * np.pi)       # Elbow heights
            pose[9][1] += 0.15 * np.sin(t * 4 * np.pi + 1)
            
            # Hand movements
            pose[10][0] += 0.1 * np.sin(t * 8 * np.pi)
            pose[11][0] += 0.1 * np.cos(t * 7 * np.pi)
            
            # Head movement
            pose[0][0] += 0.05 * np.sin(t * 3 * np.pi)
            
            sequence.append(pose)
        
        return np.array(sequence)

    def _create_bounce_sequence(self):
        """Create bouncing dance sequence"""
        base_pose = self._create_base_pose()
        sequence = []
        
        for i in range(4):
            pose = base_pose.copy()
            bounce = 0.03 * np.sin(i * np.pi / 2)
            
            # Apply bounce to whole body
            for point_idx in range(23):
                pose[point_idx][1] += bounce
            
            # Extra bounce on shoulders and hips
            pose[6][1] += bounce * 0.5   # Left shoulder
            pose[7][1] += bounce * 0.5   # Right shoulder
            pose[12][1] += bounce * 0.7  # Left hip
            pose[13][1] += bounce * 0.7  # Right hip
            
            sequence.append(pose)
        
        return np.array(sequence)

    def _create_robot_sequence(self):
        """Create robot-style dance sequence"""
        base_pose = self._create_base_pose()
        sequence = []
        
        positions = [
            # Position 1: Arms at sides
            {"arms": [(0.32, 0.48), (0.68, 0.48), (0.23, 0.68), (0.77, 0.68)]},
            # Position 2: Left arm up
            {"arms": [(0.35, 0.25), (0.68, 0.48), (0.30, 0.15), (0.77, 0.68)]},
            # Position 3: Both arms up
            {"arms": [(0.35, 0.25), (0.65, 0.25), (0.30, 0.15), (0.70, 0.15)]},
            # Position 4: Right arm up
            {"arms": [(0.32, 0.48), (0.65, 0.25), (0.23, 0.68), (0.70, 0.15)]},
        ]
        
        for pos_data in positions:
            pose = base_pose.copy()
            
            # Set arm positions (robot-like discrete positions)
            arms = pos_data["arms"]
            pose[8] = arms[0]  # Left elbow
            pose[9] = arms[1]  # Right elbow  
            pose[10] = arms[2] # Left wrist
            pose[11] = arms[3] # Right wrist
            
            # Slight head tilt for robot effect
            pose[0][0] += 0.02 * (len(sequence) % 2 * 2 - 1)
            
            sequence.append(pose)
        
        return np.array(sequence)

# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_SimpleDancePoser": BAIS1C_SimpleDancePoser}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SimpleDancePoser": "🕺 BAIS1C Simple Dance Poser"}

# Self-test function
def test_simple_dance_poser():
    """Test the Simple Dance Poser functionality"""
    poser = BAIS1C_SimpleDancePoser()
    
    # Test built-in dance creation
    styles = ["hiphop", "ballet", "freestyle", "bounce", "robot"]
    
    for style in styles:
        dance_seq = poser.built_in_dances[style]
        print(f"✅ {style.capitalize()} dance: {dance_seq.shape}")
    
    # Test pose resampling
    test_poses = poser.built_in_dances["hiphop"]
    resampled = poser._resample_poses(test_poses, 16)
    print(f"✅ Pose resampling: {test_poses.shape[0]} → {resampled.shape[0]} frames")
    
    # Test audio analysis
    sample_rate = 44100
    duration = 3.0
    test_audio = {
        "waveform": np.random.randn(int(sample_rate * duration)),
        "sample_rate": sample_rate
    }
    
    audio_features = poser._analyze_audio_simple(test_audio, 24, debug=True)
    print(f"✅ Audio analysis: {audio_features['total_frames']} frames, {audio_features['duration']:.1f}s")
    
    return True

# Uncomment to run self-test
# test_simple_dance_poser()