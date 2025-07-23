class BAIS1C_PoseExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),  # Connect from existing video loader
                "sample_stride": ("INT", {"default": 1, "min": 1, "max": 10}),
                "title": ("STRING", {"default": "extracted_poses"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "POSE")  
    RETURN_NAMES = ("pose_json_path", "pose_tensor")
    FUNCTION = "extract_poses"
    CATEGORY = "BASIC/Pose"