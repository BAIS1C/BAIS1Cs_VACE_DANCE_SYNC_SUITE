# simple_dance_poser.py
import torch, numpy as np, librosa, json, os, glob

class BAIS1C_SimpleDancePoser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "dance_source": (["library", "built_in"], {"default": "built_in"}),
                "library_dance": (cls._get_available_dances(), {"default": "none"}),
                "built_in_style": (["hiphop", "ballet", "freestyle", "bounce", "robot"], {"default": "hiphop"}),
                "dance_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "movement_smoothing": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05}),
                "music_reactivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "animation_loops": ("INT", {"default": 4, "min": 1, "max": 20}),
                "react_to": (["beat", "bass", "energy", "none"], {"default": "beat"}),
                "reaction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "output_fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "height": ("INT", {"default": 896, "min": 256, "max": 1024}),
                "visualization": (["stickman", "dots", "skeleton", "none"], {"default": "stickman"}),
                "background": (["black", "white", "dark_blue"], {"default": "black"}),
                "sync_meta": ("DICT",),  # Always present!
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_pose_tensor": ("POSE",),
            }
        }

    RETURN_TYPES = ("POSE", "IMAGE", "DICT", "STRING")
    RETURN_NAMES = ("animated_poses", "dance_video", "sync_meta", "creation_info")
    FUNCTION = "create_dance_animation"
    CATEGORY = "BAIS1C VACE Suite/Creative"

    @classmethod
    def _get_available_dances(cls):
        # ... (existing implementation, no change)
        return ["none"]  # (truncated for brevity, use previous full code)

    def __init__(self):
        # ... (existing implementation, no change)
        pass

    def create_dance_animation(self, audio, dance_source, library_dance, built_in_style,
                              dance_speed, movement_smoothing, music_reactivity, animation_loops,
                              react_to, reaction_strength, output_fps, width, height,
                              visualization, background, sync_meta, debug, input_pose_tensor=None):
        # -- existing pipeline, unchanged (use your implementation) --
        # At end, propagate meta:
        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "creation_source": dance_source,
            "fps": output_fps,
            "width": width,
            "height": height,
            "pose_count": len(input_pose_tensor) if input_pose_tensor is not None else None
        })
        creation_info = (
            f"SimpleDancePoser: {dance_source} | {built_in_style or library_dance}\n"
            f"Meta: {sync_meta_out}"
        )
        # Dummy outputs below, plug your pipeline here:
        animated_poses = torch.zeros((12,128,2))
        dance_video = torch.zeros((12,height,width,3))
        return animated_poses, dance_video, sync_meta_out, creation_info

NODE_CLASS_MAPPINGS = {"BAIS1C_SimpleDancePoser": BAIS1C_SimpleDancePoser}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SimpleDancePoser": "🕺 BAIS1C Simple Dance Poser"}

# -- Self-test --
def _test_simple_dance_poser():
    node = BAIS1C_SimpleDancePoser()
    audio = {"waveform":np.random.randn(44100*2),"sample_rate":44100}
    meta = {"fps":24,"title":"unittest"}
    res = node.create_dance_animation(
        audio,"built_in","none","robot",1.0,0.3,0.5,4,
        "beat",0.3,24,512,896,"stickman","black",meta,True,None
    )
    print(res[3])
# _test_simple_dance_poser()
