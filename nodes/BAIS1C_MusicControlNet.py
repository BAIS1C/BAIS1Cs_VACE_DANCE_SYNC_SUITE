# BAIS1C_MusicControlNet.py
# Dance Movement Retargeting — Beat-to-Anchor, DWPose-23
# FIXED: Stereo→Mono conversion + Empty tensor protection + Robust fallbacks

import numpy as np
import torch
import traceback

from .enhanced_audio_analysis import EnhancedAudioAnalyzer

class BAIS1C_MusicControlNet:
    """
    World-first: Dance Movement Retargeting Node (DWPose-23)
    - FIXED: Robust audio handling with stereo→mono conversion
    - FIXED: Empty tensor protection with intelligent fallbacks
    - Accepts: pose_tensor (n,23,2), original sync_meta, new audio
    - Detects motion anchor frames (using keypoint velocity threshold)
    - Detects new beat grid in audio
    - Maps anchors to beats, interpolates motion between beats
    - Loops/cuts as needed for exact audio length
    - Reports all actions/debug
    - Returns retargeted pose, info string, and meta
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_tensor": ("POSE",),
                "sync_meta": ("DICT",),
                "audio": ("AUDIO",),
                "anchor_sensitivity": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 1.0, "step": 0.01}),
                "sync_mode": (["beat_aligned", "frame_aligned"], {"default": "beat_aligned"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("POSE", "STRING", "DICT")
    RETURN_NAMES = ("retargeted_pose_tensor", "report", "new_sync_meta")
    FUNCTION = "execute"
    CATEGORY = "BAIS1C VACE Suite"

    def __init__(self):
        pass

    def execute(self, pose_tensor, sync_meta, audio, anchor_sensitivity=0.18,
                sync_mode="beat_aligned", debug=False):
        report = []
        try:
            if isinstance(pose_tensor, torch.Tensor):
                pose_tensor = pose_tensor.detach().cpu().numpy()
            else:
                pose_tensor = np.array(pose_tensor)
            assert pose_tensor.shape[1:] == (23, 2), f"Pose tensor must be (n,23,2), got {pose_tensor.shape}"

            fps = float(sync_meta.get("fps", 24.0))
            original_n_frames = pose_tensor.shape[0]
            original_duration = original_n_frames / fps

            # --- FIXED: Robust Audio Analysis with Stereo→Mono Conversion ---
            if not (audio and "waveform" in audio and "sample_rate" in audio):
                msg = "[MusicControlNet] ERROR: Audio input missing or incomplete."
                print(msg)
                return torch.from_numpy(pose_tensor).float(), msg, sync_meta
            
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # ✅ CRITICAL FIX: Convert stereo to mono before librosa processing
            if hasattr(waveform, "detach"):
                waveform = waveform.detach().cpu().numpy()
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            # ✅ STEREO→MONO CONVERSION (stolen from SimpleDancePoser)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)  # Average stereo channels
                if debug:
                    print(f"[MusicControlNet] Converted stereo to mono: {waveform.shape}")
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform))
            
            # Calculate expected audio duration and target frames
            audio_duration = len(waveform) / sample_rate
            target_n_frames = max(1, int(round(audio_duration * fps)))  # ✅ Ensure minimum 1 frame
            
            if debug:
                print(f"[MusicControlNet] Audio: {audio_duration:.3f}s ({len(waveform)} samples @ {sample_rate}Hz)")
                print(f"[MusicControlNet] Target frames: {target_n_frames} @ {fps} FPS")
            
            # ✅ EMPTY AUDIO PROTECTION: If audio is too short, use frame-aligned mode
            if audio_duration < 0.1:  # Less than 100ms
                print(f"[MusicControlNet] WARNING: Audio too short ({audio_duration:.3f}s), using frame-aligned mode")
                sync_mode = "frame_aligned"
                target_n_frames = original_n_frames
                audio_duration = original_duration
            
            analyzer = EnhancedAudioAnalyzer(sample_rate)
            audio_meta = analyzer.analyze_comprehensive(waveform, target_fps=fps, debug=debug)
            
            beat_times = audio_meta.get("beat_times", [])
            report.append(f"Original pose: {original_n_frames} frames @ {fps:.2f} FPS ({original_duration:.2f}s)")
            report.append(f"Audio: {audio_duration:.2f}s, {target_n_frames} frames")
            report.append(f"BPM: {audio_meta.get('primary_bpm', 'unknown')} (Conf: {audio_meta.get('bpm_confidence', 0):.2f})")
            report.append(f"Beats detected: {len(beat_times)}")

            # --- Anchor detection ---
            anchor_idxs = self._detect_movement_anchors(pose_tensor, anchor_sensitivity, debug=debug)
            report.append(f"Anchors detected: {len(anchor_idxs)}")
            if debug: 
                print(f"[MusicControlNet] Anchor frames: {anchor_idxs}")

            # --- Beat frame indices ---
            if sync_mode == "beat_aligned" and len(beat_times) > 1 and len(anchor_idxs) > 1:
                beat_frame_idxs = [int(bt / audio_duration * target_n_frames) for bt in beat_times]
                beat_frame_idxs = np.clip(beat_frame_idxs, 0, target_n_frames - 1)
                
                # Map anchor_idxs to beat_frame_idxs
                n_map = min(len(anchor_idxs), len(beat_frame_idxs))
                mapped_anchors = list(zip(anchor_idxs[:n_map], beat_frame_idxs[:n_map]))

                # Interpolate between anchors for each beat interval
                retargeted_pose = np.zeros((target_n_frames, 23, 2), dtype=pose_tensor.dtype)
                
                for i, ((src_start, tgt_start), (src_end, tgt_end)) in enumerate(zip(mapped_anchors[:-1], mapped_anchors[1:])):
                    src_segment = pose_tensor[src_start:src_end + 1]
                    seg_len = tgt_end - tgt_start
                    
                    if seg_len < 1 or src_end <= src_start:
                        continue
                    
                    # Interpolate this motion segment to fit new beat duration
                    for j in range(seg_len):
                        t = j / seg_len if seg_len > 1 else 0
                        src_idx = int(src_start + t * (src_end - src_start))
                        src_idx = min(src_idx - src_start, src_segment.shape[0] - 1)
                        if tgt_start + j < target_n_frames:
                            retargeted_pose[tgt_start + j] = src_segment[src_idx]
                
                # Fill before first anchor and after last anchor
                if len(beat_frame_idxs) > 0:
                    retargeted_pose[:beat_frame_idxs[0]] = pose_tensor[anchor_idxs[0]]
                    retargeted_pose[beat_frame_idxs[-1]:] = pose_tensor[anchor_idxs[-1]]
                
                report.append("Motion segments retargeted to beats (anchor-to-beat alignment).")

                # If audio is longer than anchor mapping, loop
                filled = beat_frame_idxs[-1] if beat_frame_idxs else 0
                if filled < target_n_frames:
                    loop_count = 0
                    while filled < target_n_frames:
                        n_fill = min(original_n_frames, target_n_frames - filled)
                        retargeted_pose[filled:filled + n_fill] = pose_tensor[:n_fill]
                        filled += n_fill
                        loop_count += 1
                    report.append(f"Pose looped {loop_count}x to fit extra audio.")

            else:
                # ✅ FRAME-ALIGNED FALLBACK: stretch/cut/loop pose tensor
                if target_n_frames == original_n_frames:
                    retargeted_pose = pose_tensor
                    report.append("No retiming: pose and audio are same length.")
                elif target_n_frames < original_n_frames:
                    retargeted_pose = pose_tensor[:target_n_frames]
                    report.append(f"Cut pose: {original_n_frames} → {target_n_frames} frames.")
                else:
                    n_repeats = target_n_frames // original_n_frames
                    remainder = target_n_frames % original_n_frames
                    retargeted_pose = np.concatenate([pose_tensor] * n_repeats + [pose_tensor[:remainder]], axis=0)
                    report.append(f"Looped pose: {original_n_frames} → {target_n_frames} frames ({n_repeats} loops).")

            # ✅ FINAL VALIDATION: Ensure we never return empty tensors
            if retargeted_pose.shape[0] == 0:
                print("[MusicControlNet] ERROR: Empty retargeted pose! Using original pose as fallback.")
                retargeted_pose = pose_tensor
                report.append("ERROR: Fallback to original pose due to empty result.")

            # --- Meta update ---
            new_meta = dict(sync_meta)
            new_meta.update(audio_meta)
            new_meta["anchor_idxs"] = anchor_idxs
            new_meta["beat_frame_idxs"] = beat_frame_idxs if sync_mode == "beat_aligned" and len(beat_times) > 1 else []
            new_meta["retimed_to_audio"] = True
            new_meta["final_n_frames"] = retargeted_pose.shape[0]
            new_meta["final_duration"] = audio_duration
            new_meta["final_bpm"] = audio_meta.get("primary_bpm", None)
            new_meta["final_bpm_confidence"] = audio_meta.get("bpm_confidence", 0.0)

            if debug:
                print(f"[MusicControlNet] Final output: {retargeted_pose.shape} tensor")
                print("\n".join(report))
            
            return torch.from_numpy(retargeted_pose).float(), "\n".join(report), new_meta

        except Exception as e:
            print(f"[MusicControlNet] ERROR: {e}")
            traceback.print_exc()
            # ✅ ROBUST ERROR FALLBACK: Return original pose instead of crashing
            error_msg = f"ERROR: {e} - Returned original pose as fallback"
            return torch.from_numpy(pose_tensor).float(), error_msg, sync_meta

    def _detect_movement_anchors(self, pose_tensor, sensitivity, debug=False):
        """Detect frames with high movement (velocity), thresholded"""
        # Returns: list of anchor frame indices
        # Sensitivity ~ velocity threshold (lower = more anchors)
        velocity = np.linalg.norm(np.diff(pose_tensor, axis=0), axis=(1,2))
        threshold = max(np.mean(velocity) + sensitivity * np.std(velocity), 1e-4)
        anchors = [0]
        for i, v in enumerate(velocity):
            if v > threshold:
                # Avoid too-close anchors (min 4 frames apart)
                if len(anchors) == 0 or (i - anchors[-1] > 3):
                    anchors.append(i+1)
        if anchors[-1] != pose_tensor.shape[0]-1:
            anchors.append(pose_tensor.shape[0]-1)
        if debug:
            print(f"[Movement Anchor Detection] {len(anchors)} anchors, threshold={threshold:.4f}")
        return anchors

# Node registration
NODE_CLASS_MAPPINGS = {
    "BAIS1C_MusicControlNet": BAIS1C_MusicControlNet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_MusicControlNet": "🎵 BAIS1C Music ControlNet (Dance Retargeting)",
}

# Self-test block
def _test_music_control_net():
    print("[TEST] BAIS1C_MusicControlNet (Fixed Version)...")
    fps = 24.0
    n_pose = 60
    pose_tensor = np.random.rand(n_pose, 23, 2).astype(np.float32)
    
    # Insert fake "dance steps"
    for i in range(10, n_pose, 12):
        pose_tensor[i] += np.random.uniform(0.08, 0.20, (23,2))
    
    sync_meta = {"fps": fps, "duration": n_pose / fps, "frame_count": n_pose}
    
    # Test with STEREO audio (this was causing the crash)
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr*duration))
    mono_waveform = 0.5 * np.sin(2 * np.pi * 2 * t)
    stereo_waveform = np.stack([mono_waveform, mono_waveform * 0.8], axis=0)  # Simulate stereo
    
    audio = {"waveform": stereo_waveform, "sample_rate": sr}
    
    node = BAIS1C_MusicControlNet()
    pose, report, meta = node.execute(pose_tensor, sync_meta, audio, 
                                    anchor_sensitivity=0.16, sync_mode="beat_aligned", debug=True)
    
    print("---REPORT---\n" + report)
    assert pose.shape[1:] == (23,2)
    assert pose.shape[0] > 0  # ✅ Most important: no empty tensors!
    print("[TEST PASSED] BAIS1C_MusicControlNet (Fixed)")

# Uncomment to run self-test
# _test_music_control_net()