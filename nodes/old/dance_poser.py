# dance_poser.py
import torch, numpy as np, librosa, json, os, glob

class BAIS1C_Suite_DancePoser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "library_dance": (cls._get_available_dances(), {"default": "none"}),
                "sync_meta": ("DICT",),  # Always sync_meta!
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_poses": ("POSE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "POSE", "DICT", "STRING")
    RETURN_NAMES = ("pose_video", "pose_tensor", "sync_meta", "report")
    FUNCTION = "sync_dance_to_music"
    CATEGORY = "BAIS1C Suite/Dance"

    @classmethod
    def _get_available_dances(cls):
        # ... (your code, as above)
        return ["none"]  # (truncated for brevity)

    def __init__(self):
        # ... (your code, as above)
        pass

    def sync_dance_to_music(self, audio, library_dance, sync_meta, debug, input_poses=None):
        # ... (pipeline unchanged, use yours)
        # At end, propagate meta:
        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "dance": library_dance,
            "frames": (input_poses.shape[0] if input_poses is not None else "unknown")
        })
        report = (
            f"DancePoser: {library_dance}\n"
            f"Meta: {sync_meta_out}"
        )
        # Dummy outputs below, plug your pipeline here:
        pose_video = torch.zeros((12,512,896,3))
        pose_tensor = torch.zeros((12,128,2))
        return pose_video, pose_tensor, sync_meta_out, report

NODE_CLASS_MAPPINGS = {"BAIS1C_Suite_DancePoser": BAIS1C_Suite_DancePoser}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_Suite_DancePoser": "🕺 BAIS1C Suite Dance Poser - Music Control Net"}

# -- Self-test --
def _test_dance_poser():
    node = BAIS1C_Suite_DancePoser()
    audio = {"waveform":np.random.randn(44100*2),"sample_rate":44100}
    meta = {"fps":24,"title":"unittest"}
    res = node.sync_dance_to_music(audio,"none",meta,True,None)
    print(res[3])
# _test_dance_poser()
