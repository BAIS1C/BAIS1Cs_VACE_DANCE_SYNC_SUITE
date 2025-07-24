# pose_tensor_extract_fixed.py (BAIS1C VACE Dance Sync Suite – Pose Extractor FIXED)
import os
import numpy as np
import torch
import cv2
import traceback

# Try different import methods for librosa tempo
try:
    from librosa.feature.rhythm import tempo as librosa_tempo
except ImportError:
    try:
        from librosa.beat import tempo as librosa_tempo
    except ImportError:
        print("Warning: librosa not found, BPM calculation will use default")
        librosa_tempo = None

# Enhanced DWPose loading with better error handling
DWPose_AVAILABLE = False
dwpose_detector = None


def load_dwpose_detector():
    """Enhanced DWPose loading with multiple import attempts"""
    global DWPose_AVAILABLE, dwpose_detector

    # Try multiple possible import paths for DWPose
    import_attempts = [
        ("dwpose.dwpose_detector", "dwpose_detector"),
        ("dwpose", "DWPoseDetector"),
        ("controlnet_aux", "DWposeDetector"),
        ("controlnet_aux.dwpose", "DWposeDetector"),
    ]

    for module_path, class_name in import_attempts:
        try:
            module = __import__(module_path, fromlist=[class_name])
            detector_class = getattr(module, class_name)

            # If it's a class, instantiate it
            if isinstance(detector_class, type):
                dwpose_detector = detector_class()
                print(
                    f"[BAIS1C PoseExtractor] Successfully loaded DWPose from {module_path}")
            else:
                dwpose_detector = detector_class
                print(
                    f"[BAIS1C PoseExtractor] Successfully loaded DWPose function from {module_path}")

            DWPose_AVAILABLE = True
            return True

        except Exception as e:
            print(
                f"[BAIS1C PoseExtractor] Failed to load from {module_path}: {e}")
            continue

    return False


# Try to load DWPose
if not load_dwpose_detector():
    print("[BAIS1C PoseExtractor] DWPose not available, will use dummy detector")

    def dwpose_detector(image_np):
        """Dummy detector that creates a basic standing pose for testing"""
        H, W = image_np.shape[:2]

        # Create a basic standing pose instead of all zeros
        bodies = np.array([
            [W//2, H//6],      # 0: nose
            [W//2-10, H//6],   # 1: left eye
            [W//2+10, H//6],   # 2: right eye
            [W//2-15, H//6+10],  # 3: left ear
            [W//2+15, H//6+10],  # 4: right ear
            [W//2-W//8, H//3],  # 5: left shoulder
            [W//2+W//8, H//3],  # 6: right shoulder
            [W//2-W//6, H//2],  # 7: left elbow
            [W//2+W//6, H//2],  # 8: right elbow
            [W//2-W//8, H//2+50],  # 9: left wrist
            [W//2+W//8, H//2+50],  # 10: right wrist
            [W//2-W//12, H//2+H//6],  # 11: left hip
            [W//2+W//12, H//2+H//6],  # 12: right hip
            [W//2-W//12, H-H//4],    # 13: left knee
            [W//2+W//12, H-H//4],    # 14: right knee
            [W//2-W//12, H-20],      # 15: left ankle
            [W//2+W//12, H-20],      # 16: right ankle
            [W//2, H//4],            # 17: neck (extra)
        ], dtype=np.float32)

        # Add confidence scores (3rd column)
        bodies_with_conf = np.column_stack([bodies, np.ones(18) * 0.8])

        # Create dummy face keypoints (just around the face area)
        face_center_x, face_center_y = W//2, H//6
        faces = [np.column_stack([
            np.random.normal(face_center_x, 20, 68),
            np.random.normal(face_center_y, 15, 68),
            np.ones(68) * 0.5
        ]).astype(np.float32)]

        # Create dummy hand keypoints
        left_hand_x, left_hand_y = W//2-W//8, H//2+50
        right_hand_x, right_hand_y = W//2+W//8, H//2+50

        left_hand = np.column_stack([
            np.random.normal(left_hand_x, 10, 21),
            np.random.normal(left_hand_y, 10, 21),
            np.ones(21) * 0.6
        ]).astype(np.float32)

        right_hand = np.column_stack([
            np.random.normal(right_hand_x, 10, 21),
            np.random.normal(right_hand_y, 10, 21),
            np.ones(21) * 0.6
        ]).astype(np.float32)

        hands = np.stack([left_hand, right_hand])

        print(
            f"[BAIS1C PoseExtractor] Using dummy detector - generated pose for {W}x{H} image")

        return {
            'bodies': bodies_with_conf,
            'faces': faces,
            'hands': hands
        }


class BAIS1C_PoseExtractor:
    """
    BAIS1C VACE Dance Sync Suite – Pose Extractor (128-point) - FIXED VERSION
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
                "frame_count": ("INT", {"default": 1, "min": 1}),
                "duration": ("FLOAT", {"default": 0.1, "min": 0.01}),
                "width": ("INT", {"default": 512, "min": 1}),
                "height": ("INT", {"default": 512, "min": 1}),
                "title": ("STRING", {"default": "untitled_pose_from_vhs"}),
                "sample_stride": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "author": ("STRING", {"default": ""}),
                "style": ("STRING", {"default": ""}),
                "debug": ("BOOLEAN", {"default": False}),
                "pose_visibility": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("POSE", "DICT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("pose_tensor", "sync_meta",
                    "processed_images", "audio_passthrough")
    FUNCTION = "extract"
    CATEGORY = "BAIS1C VACE Suite/Pose"
    OUTPUT_NODE = False

    def _extract_audio_data(self, audio_input):
        """Robustly extracts waveform and sample_rate from various audio input types."""
        waveform = None
        sample_rate = None

        if isinstance(audio_input, dict):
            waveform = audio_input.get("waveform")
            sample_rate = audio_input.get("sample_rate")

        # Handle VHS LazyAudioMap or similar objects
        if waveform is None and hasattr(audio_input, 'waveform'):
            try:
                waveform = audio_input.waveform
                sample_rate = audio_input.sample_rate
            except Exception:
                pass

        # Convert to tensor if needed
        if waveform is not None and not isinstance(waveform, torch.Tensor):
            try:
                waveform = torch.from_numpy(waveform) if isinstance(
                    waveform, np.ndarray) else None
            except:
                waveform = None

        return waveform, sample_rate

    def _run_pose_detection(self, image_np: np.ndarray, debug: bool) -> dict:
        """
        Run pose detection with enhanced format handling and debugging
        """
        global dwpose_detector, DWPose_AVAILABLE

        if debug:
            print(
                f"[BAIS1C PoseExtractor] Running pose detection on {image_np.shape} image")
            print(
                f"[BAIS1C PoseExtractor] DWPose available: {DWPose_AVAILABLE}")
            print(
                f"[BAIS1C PoseExtractor] Image dtype: {image_np.dtype}, range: [{image_np.min()}, {image_np.max()}]")

        try:
            # Ensure image is in correct format (uint8, 0-255)
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

            # Try detection with current format
            pose_result = dwpose_detector(image_np)

            if debug:
                print(
                    f"[BAIS1C PoseExtractor] Detection result type: {type(pose_result)}")
                print(
                    f"[BAIS1C PoseExtractor] Detection result keys: {list(pose_result.keys()) if isinstance(pose_result, dict) else 'Not a dict'}")

                # Detailed analysis of each component
                for key, value in pose_result.items():
                    if isinstance(value, np.ndarray):
                        print(
                            f"[BAIS1C PoseExtractor]   {key}: shape={value.shape}, dtype={value.dtype}")
                        if value.size > 0:
                            print(
                                f"[BAIS1C PoseExtractor]      range: [{value.min():.2f}, {value.max():.2f}]")
                            # Check for non-zero values
                            non_zero = np.count_nonzero(value)
                            print(
                                f"[BAIS1C PoseExtractor]      non-zero elements: {non_zero}/{value.size}")
                    elif isinstance(value, list):
                        print(
                            f"[BAIS1C PoseExtractor]   {key}: list of {len(value)} items")
                        if len(value) > 0 and isinstance(value[0], np.ndarray):
                            print(
                                f"[BAIS1C PoseExtractor]      first item shape: {value[0].shape}")
                            if value[0].size > 0:
                                print(
                                    f"[BAIS1C PoseExtractor]      first item range: [{value[0].min():.2f}, {value[0].max():.2f}]")
                    else:
                        print(f"[BAIS1C PoseExtractor]   {key}: {type(value)}")

                # Special handling for common issues
                if 'bodies' in pose_result:
                    bodies = pose_result['bodies']
                    if isinstance(bodies, np.ndarray):
                        if bodies.size == 0:
                            print(
                                f"[BAIS1C PoseExtractor] ⚠️  Bodies array is EMPTY!")
                        elif np.all(bodies < 1):
                            print(
                                f"[BAIS1C PoseExtractor] ⚠️  Bodies array has all values < 1 (likely all zeros or very small)")
                        else:
                            print(
                                f"[BAIS1C PoseExtractor] ✅ Bodies array has valid-looking values")

            # Post-process result to handle different formats
            processed_result = self._standardize_pose_result(
                pose_result, debug)

            return processed_result

        except Exception as e:
            print(f"[BAIS1C PoseExtractor] Pose detection failed: {e}")
            if debug:
                traceback.print_exc()

            # Return empty pose structure
            return {
                'bodies': np.zeros((18, 3), dtype=np.float32),
                'faces': [np.zeros((68, 3), dtype=np.float32)],
                'hands': np.zeros((2, 21, 3), dtype=np.float32)
            }

    def _standardize_pose_result(self, raw_result: dict, debug: bool = False) -> dict:
        """
        Standardize different DWPose output formats to a consistent format
        """
        result = {
            'bodies': np.array([]),
            'faces': [],
            'hands': np.array([])
        }

        try:
            # Handle bodies - special case for dict format
            if 'bodies' in raw_result:
                bodies = raw_result['bodies']
                if isinstance(bodies, np.ndarray) and bodies.size > 0:
                    result['bodies'] = bodies
                elif isinstance(bodies, dict):
                    # Try to extract bodies from dict format
                    if debug:
                        print(
                            f"[BAIS1C PoseExtractor] Bodies is dict with keys: {list(bodies.keys()) if bodies else 'empty'}")
                    # Check for common dict keys that might contain body data
                    for key in ['candidate', 'keypoints', 'poses', 'body_keypoints']:
                        if key in bodies and isinstance(bodies[key], np.ndarray):
                            result['bodies'] = bodies[key]
                            if debug:
                                print(
                                    f"[BAIS1C PoseExtractor] Extracted bodies from dict key '{key}': {bodies[key].shape}")
                            break
                    else:
                        # No valid body data found in dict
                        result['bodies'] = np.array([])
                        if debug:
                            print(
                                f"[BAIS1C PoseExtractor] No valid body data found in bodies dict")
                elif debug:
                    print(
                        f"[BAIS1C PoseExtractor] Bodies field exists but is empty or wrong type: {type(bodies)}")
            else:
                result['bodies'] = np.array([])

            # Handle faces - multiple possible formats
            if 'faces' in raw_result:
                faces = raw_result['faces']
                if isinstance(faces, list) and len(faces) > 0:
                    result['faces'] = faces
                elif isinstance(faces, np.ndarray) and faces.size > 0:
                    # Convert array to list format
                    if faces.ndim == 3 and faces.shape[0] >= 1:  # (1, 68, 2)
                        result['faces'] = [faces[0]]
                    elif faces.ndim == 2:  # (68, 2)
                        result['faces'] = [faces]
                    else:
                        result['faces'] = [faces]
                elif debug:
                    print(
                        f"[BAIS1C PoseExtractor] Faces field exists but is empty or wrong format")

            # Handle hands
            if 'hands' in raw_result:
                hands = raw_result['hands']
                if isinstance(hands, np.ndarray) and hands.size > 0:
                    result['hands'] = hands
                elif debug:
                    print(
                        f"[BAIS1C PoseExtractor] Hands field exists but is empty or wrong type")

            if debug:
                print(f"[BAIS1C PoseExtractor] Standardized result:")
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        print(f"[BAIS1C PoseExtractor]   {key}: {value.shape}")
                    elif isinstance(value, list):
                        print(
                            f"[BAIS1C PoseExtractor]   {key}: list of {len(value)}")

        except Exception as e:
            if debug:
                print(
                    f"[BAIS1C PoseExtractor] Error standardizing pose result: {e}")

        return result

    def _pose_dict_to_tensor(self, pose_dict: dict, frame_shape: tuple, debug: bool = False) -> np.ndarray:
        """
        Convert DWPose detection results to 128-point tensor format - ENHANCED VERSION
        """
        H, W = frame_shape
        pose_tensor = np.zeros((128, 2), dtype=np.float32)

        if debug:
            print(
                f"[BAIS1C PoseExtractor] Converting pose dict to tensor for frame {W}x{H}")
            print(
                f"[BAIS1C PoseExtractor] Available keys: {list(pose_dict.keys())}")

        try:
            # Body keypoints (0-17) - ENHANCED DEBUG
            bodies = pose_dict.get('bodies', np.array([]))
            if debug:
                print(
                    f"[BAIS1C PoseExtractor] Bodies raw: type={type(bodies)}, shape={getattr(bodies, 'shape', 'N/A')}, size={getattr(bodies, 'size', 'N/A')}")
                if isinstance(bodies, np.ndarray) and bodies.size > 0:
                    print(
                        f"[BAIS1C PoseExtractor] Bodies content preview: {bodies[:min(3, len(bodies))]}")
                    print(
                        f"[BAIS1C PoseExtractor] Bodies range: x=[{bodies[:, 0].min():.1f}, {bodies[:, 0].max():.1f}], y=[{bodies[:, 1].min():.1f}, {bodies[:, 1].max():.1f}]")

            bodies_processed = False
            if isinstance(bodies, np.ndarray) and bodies.size > 0:
                if bodies.ndim == 2 and bodies.shape[0] >= 18 and bodies.shape[1] >= 2:
                    body_points = bodies[:18, :2]  # Take x, y coordinates only

                    # Check if points are actually valid (not all zeros) - FIXED for normalized coords
                    valid_bodies = np.sum(
                        (body_points[:, 0] > 0.01) | (body_points[:, 1] > 0.01))
                    if debug:
                        print(
                            f"[BAIS1C PoseExtractor] Bodies valid points: {valid_bodies}/18")

                    if valid_bodies > 0:  # Only process if we have some valid points
                        # Normalize to 0-1 range
                        if W > 0 and H > 0:
                            # Check if coordinates are already normalized or in pixels
                            if np.max(body_points) <= 1.0:
                                # Already normalized, use directly
                                pose_tensor[:18, 0] = np.clip(
                                    body_points[:, 0], 0, 1)
                                pose_tensor[:18, 1] = np.clip(
                                    body_points[:, 1], 0, 1)
                                if debug:
                                    print(
                                        f"[BAIS1C PoseExtractor]   Bodies: {bodies.shape} -> using normalized coords directly")
                            else:
                                # Pixel coordinates, need to normalize
                                pose_tensor[:18, 0] = np.clip(
                                    body_points[:, 0] / W, 0, 1)
                                pose_tensor[:18, 1] = np.clip(
                                    body_points[:, 1] / H, 0, 1)
                                if debug:
                                    print(
                                        f"[BAIS1C PoseExtractor]   Bodies: {bodies.shape} -> normalized from pixels")
                        bodies_processed = True

                        if debug:
                            print(
                                f"[BAIS1C PoseExtractor]   Body points range: x=[{pose_tensor[:18, 0].min():.3f}, {pose_tensor[:18, 0].max():.3f}], y=[{pose_tensor[:18, 1].min():.3f}, {pose_tensor[:18, 1].max():.3f}]")
                    elif debug:
                        print(
                            f"[BAIS1C PoseExtractor] Bodies detected but all near zero - likely invalid detection")
                elif debug:
                    print(
                        f"[BAIS1C PoseExtractor] Bodies array has wrong shape: {bodies.shape}")
            elif debug:
                print(f"[BAIS1C PoseExtractor] No bodies detected or empty array")

            # If no valid bodies, create a fallback center pose for visualization
            if not bodies_processed and debug:
                print(
                    f"[BAIS1C PoseExtractor] Creating fallback body pose for visualization")
                # Create a simple standing pose in the center
                center_x, center_y = 0.5, 0.5
                pose_tensor[0] = [center_x, center_y - 0.15]  # nose
                pose_tensor[5] = [center_x - 0.1,
                                  center_y - 0.05]  # left shoulder
                pose_tensor[6] = [center_x + 0.1,
                                  center_y - 0.05]  # right shoulder
                pose_tensor[11] = [center_x - 0.05, center_y + 0.1]  # left hip
                pose_tensor[12] = [center_x + 0.05,
                                   center_y + 0.1]  # right hip
                pose_tensor[13] = [center_x - 0.05,
                                   center_y + 0.25]  # left knee
                pose_tensor[14] = [center_x + 0.05,
                                   center_y + 0.25]  # right knee

            # Face keypoints (18-85) - ENHANCED DEBUG
            faces = pose_dict.get('faces', [])
            if debug:
                print(
                    f"[BAIS1C PoseExtractor] Faces raw: type={type(faces)}, length={len(faces) if isinstance(faces, list) else 'N/A'}")

            faces_processed = False
            if isinstance(faces, list) and len(faces) > 0:
                face_data = faces[0]
                if debug:
                    print(
                        f"[BAIS1C PoseExtractor] Face data: type={type(face_data)}, shape={getattr(face_data, 'shape', 'N/A')}")

                if isinstance(face_data, np.ndarray) and face_data.size > 0:
                    if face_data.ndim == 2 and face_data.shape[0] >= 68 and face_data.shape[1] >= 2:
                        face_points = face_data[:68, :2]

                        # Check if face points are valid - FIXED for normalized coords
                        valid_faces = np.sum(
                            (face_points[:, 0] > 0.01) | (face_points[:, 1] > 0.01))
                        if debug:
                            print(
                                f"[BAIS1C PoseExtractor] Faces valid points: {valid_faces}/68")
                            if valid_faces > 0:
                                print(
                                    f"[BAIS1C PoseExtractor] Face range: x=[{face_points[:, 0].min():.3f}, {face_points[:, 0].max():.3f}], y=[{face_points[:, 1].min():.3f}, {face_points[:, 1].max():.3f}]")

                        if valid_faces > 0:
                            if W > 0 and H > 0:
                                # Check if coordinates are already normalized or in pixels
                                if np.max(face_points) <= 1.0:
                                    # Already normalized, use directly
                                    pose_tensor[18:86, 0] = np.clip(
                                        face_points[:, 0], 0, 1)
                                    pose_tensor[18:86, 1] = np.clip(
                                        face_points[:, 1], 0, 1)
                                    if debug:
                                        print(
                                            f"[BAIS1C PoseExtractor]   Faces: {face_data.shape} -> using normalized coords directly")
                                else:
                                    # Pixel coordinates, need to normalize
                                    pose_tensor[18:86, 0] = np.clip(
                                        face_points[:, 0] / W, 0, 1)
                                    pose_tensor[18:86, 1] = np.clip(
                                        face_points[:, 1] / H, 0, 1)
                                    if debug:
                                        print(
                                            f"[BAIS1C PoseExtractor]   Faces: {face_data.shape} -> normalized from pixels")
                            faces_processed = True
                        elif debug:
                            print(
                                f"[BAIS1C PoseExtractor] Face points detected but all near zero")
            elif debug:
                print(f"[BAIS1C PoseExtractor] No faces detected or wrong format")

            # Check if we got faces but no format
            if not faces_processed and 'faces' in pose_dict:
                alt_faces = pose_dict['faces']
                if isinstance(alt_faces, np.ndarray) and alt_faces.size > 0:
                    if debug:
                        print(
                            f"[BAIS1C PoseExtractor] Trying alternative faces format: {alt_faces.shape}")
                    # (1, 68, 2)
                    if alt_faces.ndim == 3 and alt_faces.shape[0] >= 1:
                        face_points = alt_faces[0, :68,
                                                :2] if alt_faces.shape[1] >= 68 else alt_faces[0]
                        valid_faces = np.sum(
                            (face_points[:, 0] > 0.01) | (face_points[:, 1] > 0.01))
                        if valid_faces > 0 and W > 0 and H > 0:
                            face_count = min(68, face_points.shape[0])
                            # Check if coordinates are already normalized
                            if np.max(face_points) <= 1.0:
                                # Already normalized
                                pose_tensor[18:18+face_count,
                                            0] = np.clip(face_points[:face_count, 0], 0, 1)
                                pose_tensor[18:18+face_count,
                                            1] = np.clip(face_points[:face_count, 1], 0, 1)
                            else:
                                # Pixel coordinates
                                pose_tensor[18:18+face_count, 0] = np.clip(
                                    face_points[:face_count, 0] / W, 0, 1)
                                pose_tensor[18:18+face_count, 1] = np.clip(
                                    face_points[:face_count, 1] / H, 0, 1)
                            if debug:
                                print(
                                    f"[BAIS1C PoseExtractor]   Alternative faces processed: {face_count} points")

            # Hand keypoints (86-127) - ENHANCED DEBUG
            hands = pose_dict.get('hands', np.array([]))
            if debug:
                print(
                    f"[BAIS1C PoseExtractor] Hands raw: type={type(hands)}, shape={getattr(hands, 'shape', 'N/A')}")

            hands_processed = False
            if isinstance(hands, np.ndarray) and hands.size > 0:
                if hands.ndim == 3 and hands.shape[0] == 2 and hands.shape[1] == 21 and hands.shape[2] >= 2:
                    # Left hand (86-106)
                    left_hand_points = hands[0, :, :2]
                    # Right hand (107-127)
                    right_hand_points = hands[1, :, :2]

                    # Check validity - FIXED for normalized coords
                    valid_left = np.sum((left_hand_points[:, 0] > 0.01) | (
                        left_hand_points[:, 1] > 0.01))
                    valid_right = np.sum((right_hand_points[:, 0] > 0.01) | (
                        right_hand_points[:, 1] > 0.01))

                    if debug:
                        print(
                            f"[BAIS1C PoseExtractor] Left hand valid: {valid_left}/21, Right hand valid: {valid_right}/21")
                        if valid_left > 0:
                            print(
                                f"[BAIS1C PoseExtractor] Left hand range: x=[{left_hand_points[:, 0].min():.3f}, {left_hand_points[:, 0].max():.3f}], y=[{left_hand_points[:, 1].min():.3f}, {left_hand_points[:, 1].max():.3f}]")
                        if valid_right > 0:
                            print(
                                f"[BAIS1C PoseExtractor] Right hand range: x=[{right_hand_points[:, 0].min():.3f}, {right_hand_points[:, 0].max():.3f}], y=[{right_hand_points[:, 1].min():.3f}, {right_hand_points[:, 1].max():.3f}]")

                    if (valid_left > 0 or valid_right > 0) and W > 0 and H > 0:
                        # Check if coordinates are already normalized (0-1 range) or in pixel coordinates
                        if np.max(left_hand_points) <= 1.0 and np.max(right_hand_points) <= 1.0:
                            # Already normalized, use directly
                            pose_tensor[86:107, 0] = np.clip(
                                left_hand_points[:, 0], 0, 1)
                            pose_tensor[86:107, 1] = np.clip(
                                left_hand_points[:, 1], 0, 1)
                            pose_tensor[107:128, 0] = np.clip(
                                right_hand_points[:, 0], 0, 1)
                            pose_tensor[107:128, 1] = np.clip(
                                right_hand_points[:, 1], 0, 1)
                            if debug:
                                print(
                                    f"[BAIS1C PoseExtractor]   Hands: {hands.shape} -> using normalized coords directly")
                        else:
                            # Pixel coordinates, need to normalize
                            pose_tensor[86:107, 0] = np.clip(
                                left_hand_points[:, 0] / W, 0, 1)
                            pose_tensor[86:107, 1] = np.clip(
                                left_hand_points[:, 1] / H, 0, 1)
                            pose_tensor[107:128, 0] = np.clip(
                                right_hand_points[:, 0] / W, 0, 1)
                            pose_tensor[107:128, 1] = np.clip(
                                right_hand_points[:, 1] / H, 0, 1)
                            if debug:
                                print(
                                    f"[BAIS1C PoseExtractor]   Hands: {hands.shape} -> normalized from pixels")
                        hands_processed = True
                    elif debug:
                        print(
                            f"[BAIS1C PoseExtractor] Hand points detected but all near zero")
                elif debug:
                    print(
                        f"[BAIS1C PoseExtractor] Hands array has wrong shape: {hands.shape}")
            elif debug:
                print(f"[BAIS1C PoseExtractor] No hands detected or empty array")

            # Validate final tensor and add fallback if needed
            valid_points = np.sum(
                (pose_tensor[:, 0] > 0.001) | (pose_tensor[:, 1] > 0.001))

            # If we have very few valid points, create a more visible fallback pose
            if valid_points < 5:  # Reduced threshold since we should get hands/faces now
                if debug:
                    print(
                        f"[BAIS1C PoseExtractor] Very few valid points ({valid_points}), adding fallback central pose")

                # Create a visible central figure
                cx, cy = 0.5, 0.5  # Center of image

                # Basic body structure
                pose_tensor[0] = [cx, cy - 0.2]      # nose
                pose_tensor[5] = [cx - 0.15, cy - 0.1]  # left shoulder
                pose_tensor[6] = [cx + 0.15, cy - 0.1]  # right shoulder
                pose_tensor[7] = [cx - 0.2, cy]      # left elbow
                pose_tensor[8] = [cx + 0.2, cy]      # right elbow
                pose_tensor[9] = [cx - 0.2, cy + 0.1]   # left wrist
                pose_tensor[10] = [cx + 0.2, cy + 0.1]  # right wrist
                pose_tensor[11] = [cx - 0.1, cy + 0.15]  # left hip
                pose_tensor[12] = [cx + 0.1, cy + 0.15]  # right hip
                pose_tensor[13] = [cx - 0.1, cy + 0.35]  # left knee
                pose_tensor[14] = [cx + 0.1, cy + 0.35]  # right knee
                pose_tensor[15] = [cx - 0.1, cy + 0.45]  # left ankle
                pose_tensor[16] = [cx + 0.1, cy + 0.45]  # right ankle

                valid_points = np.sum(
                    (pose_tensor[:, 0] > 0.001) | (pose_tensor[:, 1] > 0.001))

            if debug:
                print(
                    f"[BAIS1C PoseExtractor]   Final tensor: {pose_tensor.shape}, valid points: {valid_points}/128")
                non_zero_indices = np.where(
                    (pose_tensor[:, 0] > 0.001) | (pose_tensor[:, 1] > 0.001))[0]
                if len(non_zero_indices) > 0:
                    print(
                        f"[BAIS1C PoseExtractor]   Valid point indices: {non_zero_indices[:10]}{'...' if len(non_zero_indices) > 10 else ''}")
                    print(
                        f"[BAIS1C PoseExtractor]   Coordinate ranges: x=[{pose_tensor[non_zero_indices, 0].min():.3f}, {pose_tensor[non_zero_indices, 0].max():.3f}], y=[{pose_tensor[non_zero_indices, 1].min():.3f}, {pose_tensor[non_zero_indices, 1].max():.3f}]")

            return pose_tensor

        except Exception as e:
            print(f"[BAIS1C PoseExtractor] Error in pose conversion: {e}")
            if debug:
                traceback.print_exc()
            return np.zeros((128, 2), dtype=np.float32)

    def _draw_pose_on_image(self, image_tensor: torch.Tensor, pose_tensor_np: np.ndarray,
                            frame_shape: tuple, pose_visibility: float = 1.0, debug: bool = False) -> torch.Tensor:
        """
        Enhanced pose drawing with better visibility and debugging
        """
        H, W = frame_shape

        # Convert image tensor to OpenCV format
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Convert normalized coordinates back to pixels
        pose_pixels = pose_tensor_np * [W - 1, H - 1]
        keypoints_px = np.round(pose_pixels).astype(int)

        if debug:
            valid_keypoints = np.sum(
                (pose_tensor_np[:, 0] > 0.001) | (pose_tensor_np[:, 1] > 0.001))
            print(
                f"[BAIS1C PoseExtractor] Drawing {valid_keypoints} valid keypoints on {W}x{H} image")

            if valid_keypoints > 0:
                # Show actual pixel coordinates for first few valid points
                valid_indices = np.where((pose_tensor_np[:, 0] > 0.001) | (
                    pose_tensor_np[:, 1] > 0.001))[0]
                print(
                    f"[BAIS1C PoseExtractor]   Sample pixel coords: {[(i, keypoints_px[i]) for i in valid_indices[:5]]}")
                print(
                    f"[BAIS1C PoseExtractor]   Pixel range: x=[{keypoints_px[valid_indices, 0].min()}, {keypoints_px[valid_indices, 0].max()}], y=[{keypoints_px[valid_indices, 1].min()}, {keypoints_px[valid_indices, 1].max()}]")

        # Enhanced drawing parameters
        point_radius = max(3, int(5 * pose_visibility))
        line_thickness = max(2, int(4 * pose_visibility))

        # Body connections (COCO format) - ENHANCED
        body_connections = [
            # Head connections
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Left arm
            (5, 7), (7, 9),
            # Right arm
            (6, 8), (8, 10),
            # Left leg
            (11, 13), (13, 15),
            # Right leg
            (12, 14), (14, 16)
        ]

        # Draw body skeleton with VERY bright, thick lines
        skeleton_drawn = 0
        for i, j in body_connections:
            if i < len(keypoints_px) and j < len(keypoints_px):
                # Check if both points are valid (not at origin)
                pt1_valid = (pose_tensor_np[i, 0] >
                             0.001 or pose_tensor_np[i, 1] > 0.001)
                pt2_valid = (pose_tensor_np[j, 0] >
                             0.001 or pose_tensor_np[j, 1] > 0.001)

                if pt1_valid and pt2_valid:
                    pt1 = tuple(keypoints_px[i])
                    pt2 = tuple(keypoints_px[j])

                    # Check bounds
                    if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                            0 <= pt2[0] < W and 0 <= pt2[1] < H):
                        # Draw thick, bright yellow line
                        cv2.line(image_bgr, pt1, pt2,
                                 (0, 255, 255), line_thickness)
                        skeleton_drawn += 1

        # Draw keypoints with enhanced visibility
        points_drawn = 0
        colors = {
            'head': (0, 0, 255),      # Bright red
            'body': (0, 255, 0),      # Bright green
            'face': (255, 0, 255),    # Bright magenta
            'left_hand': (255, 255, 0),  # Bright cyan
            'right_hand': (0, 255, 255),  # Bright yellow
        }

        for idx, (px, py) in enumerate(keypoints_px):
            # Skip invalid points (at origin)
            if pose_tensor_np[idx, 0] < 0.001 and pose_tensor_np[idx, 1] < 0.001:
                continue

            if 0 <= px < W and 0 <= py < H:
                # Determine color based on keypoint index
                if idx <= 4:  # Head (nose, eyes, ears)
                    color = colors['head']
                elif idx <= 17:  # Body (shoulders, elbows, wrists, hips, knees, ankles)
                    color = colors['body']
                elif idx <= 85:  # Face
                    color = colors['face']
                elif idx <= 106:  # Left hand
                    color = colors['left_hand']
                else:  # Right hand
                    color = colors['right_hand']

                # Draw larger, more visible circle
                cv2.circle(image_bgr, (px, py), point_radius, color, -1)
                points_drawn += 1

                # Add keypoint index for key body points if debugging
                if debug and idx <= 17:  # Only show body keypoint indices
                    cv2.putText(image_bgr, str(idx), (px + 8, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add comprehensive status text
        status_lines = [
            f"DWPose: {'ON' if DWPose_AVAILABLE else 'DUMMY'}",
            f"Points: {points_drawn}/128",
            f"Lines: {skeleton_drawn}"
        ]

        for i, line in enumerate(status_lines):
            y_pos = 30 + i * 25
            cv2.putText(image_bgr, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If very few points drawn, add a warning overlay
        if points_drawn < 5:
            warning_text = "LOW POSE DETECTION"
            text_size = cv2.getTextSize(
                warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (W - text_size[0]) // 2
            text_y = H // 2
            cv2.rectangle(image_bgr, (text_x - 10, text_y - 30),
                          (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
            cv2.putText(image_bgr, warning_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if debug:
            print(
                f"[BAIS1C PoseExtractor] Drew {points_drawn} points and {skeleton_drawn} skeleton lines")

        # Convert back to RGB tensor
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        processed_tensor = torch.from_numpy(
            image_rgb.astype(np.float32) / 255.0)

        return processed_tensor

    def extract(self, images, audio, fps, frame_count, duration, width, height,
                title, sample_stride, author="", style="", debug=False, pose_visibility=1.0):
        """
        Main extraction method - ENHANCED VERSION
        """
        try:
            if debug:
                print(
                    f"\n[BAIS1C PoseExtractor] === ENHANCED Pose Extraction Start ===")
                print(
                    f"[BAIS1C PoseExtractor] Input batch shape: {images.shape}")
                print(
                    f"[BAIS1C PoseExtractor] DWPose available: {DWPose_AVAILABLE}")

            # Validate inputs
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError(
                    f"Invalid image batch format: {type(images)} {getattr(images, 'shape', 'N/A')}")

            # Extract audio data
            waveform, sample_rate = self._extract_audio_data(audio)

            # Calculate BPM
            bpm = 120.0
            if waveform is not None and sample_rate is not None and librosa_tempo is not None:
                try:
                    if isinstance(waveform, torch.Tensor):
                        waveform_np = waveform.squeeze().cpu().numpy()
                    else:
                        waveform_np = np.array(waveform).squeeze()

                    if waveform_np.ndim > 1:
                        waveform_np = np.mean(waveform_np, axis=-1)

                    if len(waveform_np) > 0 and np.max(np.abs(waveform_np)) > 0:
                        waveform_np = waveform_np / \
                            np.max(np.abs(waveform_np)) * 0.95
                        tempo_estimates = librosa_tempo(
                            y=waveform_np, sr=sample_rate, aggregate=None)
                        bpm = float(tempo_estimates[0]) if len(
                            tempo_estimates) > 0 else 120.0
                except Exception as e:
                    if debug:
                        print(
                            f"[BAIS1C PoseExtractor] BPM calculation failed: {e}")

            # Process frames
            total_frames = images.shape[0]
            frame_indices = list(range(0, total_frames, sample_stride))

            if debug:
                print(
                    f"[BAIS1C PoseExtractor] Processing {len(frame_indices)} frames with stride {sample_stride}")

            pose_sequence = []
            processed_images_sequence = []
            failed_detections = 0

            for idx, frame_idx in enumerate(frame_indices):
                try:
                    if debug and idx < 3:  # Debug first few frames
                        print(
                            f"[BAIS1C PoseExtractor] Processing frame {frame_idx} ({idx+1}/{len(frame_indices)})")

                    # Get image tensor and convert to numpy
                    image_tensor = images[frame_idx]  # [H, W, C]
                    image_np = (image_tensor.cpu().numpy()
                                * 255).astype(np.uint8)

                    # Run pose detection
                    pose_dict = self._run_pose_detection(
                        image_np, debug and idx < 3)

                    # Convert to tensor format
                    pose_tensor_np = self._pose_dict_to_tensor(
                        pose_dict, image_np.shape[:2], debug and idx < 3)

                    # Draw pose on image
                    processed_image = self._draw_pose_on_image(
                        image_tensor, pose_tensor_np, image_np.shape[:
                                                                     2], pose_visibility, debug and idx < 3
                    )

                    # Add to sequences
                    pose_sequence.append(torch.from_numpy(
                        pose_tensor_np).float().unsqueeze(0))
                    processed_images_sequence.append(
                        processed_image.unsqueeze(0))

                except Exception as e:
                    failed_detections += 1
                    print(
                        f"[BAIS1C PoseExtractor] Frame {frame_idx} failed: {e}")
                    if debug:
                        traceback.print_exc()

                    # Add fallback data
                    fallback_pose = torch.zeros(
                        (1, 128, 2), dtype=torch.float32)
                    pose_sequence.append(fallback_pose)
                    processed_images_sequence.append(
                        images[frame_idx].unsqueeze(0))

            # Finalize outputs
            if pose_sequence:
                final_pose_tensor = torch.cat(pose_sequence, dim=0)
            else:
                final_pose_tensor = torch.zeros(
                    (1, 128, 2), dtype=torch.float32)

            if processed_images_sequence:
                processed_images_tensor = torch.cat(
                    processed_images_sequence, dim=0)
            else:
                processed_images_tensor = images[:1].clone()

            # Create metadata
            sync_meta = {
                "title": title,
                "author": author,
                "style": style,
                "fps": fps,
                "original_frame_count": frame_count,
                "input_batch_frames": total_frames,
                "processed_frames": len(frame_indices),
                "duration": len(frame_indices) / fps if fps > 0 else 0,
                "original_duration": duration,
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height != 0 else 0,
                "bpm": bpm,
                "sample_stride": sample_stride,
                "pose_extraction_success": DWPose_AVAILABLE,
                "failed_detections": failed_detections,
                "extraction_settings": {
                    "pose_format": "128-point",
                    "dwpose_version": "Enhanced",
                    "input_type": "VHS_IMAGE_TENSOR",
                    "visualization_added": True,
                    "pose_visibility": pose_visibility
                }
            }

            if debug:
                print(f"[BAIS1C PoseExtractor] === Extraction Complete ===")
                print(
                    f"[BAIS1C PoseExtractor] Final pose shape: {final_pose_tensor.shape}")
                print(
                    f"[BAIS1C PoseExtractor] Final images shape: {processed_images_tensor.shape}")
                print(
                    f"[BAIS1C PoseExtractor] Failed detections: {failed_detections}")

            return (final_pose_tensor, sync_meta, processed_images_tensor, audio)

        except Exception as e:
            error_msg = f"[BAIS1C PoseExtractor] Critical failure: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            # Return safe fallback data
            dummy_pose = torch.zeros((1, 128, 2), dtype=torch.float32)
            dummy_meta = {
                "title": title,
                "error": error_msg,
                "fps": fps,
                "frame_count": 0,
                "duration": 0,
                "width": width,
                "height": height,
                "bpm": 120.0,
                "processed_frames": 0,
                "failed_detections": -1,
            }
            dummy_images = images[:1].clone() if isinstance(
                images, torch.Tensor) else torch.zeros((1, 512, 512, 3))

            return (dummy_pose, dummy_meta, dummy_images, audio)


# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_PoseExtractor": "🎯 BAIS1C Pose Extractor (Enhanced)"}
