🕺 BAIS1C VACE Dance Sync Suite (BETA STILL NEEDS FIXES)
Professional ComfyUI toolkit for pose extraction, dance library management, and music-driven dance animation.
Pipeline: video-to-pose extraction, library export, music-reactive dance generation, and frame-accurate BPM synchronization.

🚀 Installation (Windows/ComfyUI)
1. Clone or Download This Folder

From your ComfyUI root, navigate to custom_nodes.

Download or clone the suite into custom_nodes\BAIS1Cs_VACE_DANCE_SYNC_SUITE\.

2. Install Required Python Packages

Open a terminal or Anaconda Prompt as Administrator.

Run:

nginx
Copy
Edit
pip install -r custom_nodes\BAIS1Cs_VACE_DANCE_SYNC_SUITE\requirements.txt
3. Download Required DWPose Models

Download the following ONNX model files:

dw-ll_ucoco_384.onnx

yolox_l.onnx

Place both files in:

Copy
Edit
ComfyUI\models\dwpose\
(Create this folder if it does not exist. You may use a different folder by setting the DWPose environment variable, but this is the default and recommended path.)

📋 Video & Pose Model Requirements
Aspect ratios:

16:9 (landscape)

9:16 (portrait)

Recommended resolutions:

Landscape: 832×468

Portrait: 468×832

Important: Using other aspect ratios will distort skeletons and break pose accuracy.

🎭 Node Overview
🎥 BAIS1C Source Video Loader
Category: BAIS1C VACE Suite/Source

Loads video, extracts audio, BPM, FPS, and metadata for the sync pipeline.

Outputs: video object, audio object, fps, bpm, frame count, duration, sync_meta dictionary.

🎯 BAIS1C Pose Extractor (128pt)
Category: BAIS1C VACE Suite/Extraction

Extracts 128-point pose tensors from video using DWPose.

Saves as JSON for reuse and library-building.

Format: 23 body, 68 face, 21 left hand, 16 right hand (all normalized).

Outputs: pose tensor (TENSOR), sync_meta (DICT).

🎵 BAIS1C Music Control Net (Auto-Sync)
Category: BAIS1C VACE Suite/Control

Syncs pose libraries to any music source using professional BPM algorithms.

Supports multiple sync/loop modes.

Outputs: synced pose tensor, pose video (visualization), sync report.

🕺 BAIS1C Simple Dance Poser (Creative Playground)
Category: BAIS1C VACE Suite/Creative

Generate creative dance animations with user-tunable style, speed, and music response.

Outputs: animated pose tensor, video, creation info.

📂 File Structure
Model files:
ComfyUI\models\dwpose\

Dance library:
custom_nodes\BAIS1Cs_VACE_DANCE_SYNC_SUITE\dance_library\
(fallback: ComfyUI\output\dance_library\)

Starter dances:

basic_walking.json

arms_up_body_sway.json

hands_on_hips.json

🔄 Example Workflows (COMING SOON)
Professional Sync Pipeline
Source Video Loader: Load video and extract BPM/metadata

Pose Extractor: Extract pose tensors and save JSON

Music Control Net: Load JSON and sync poses to any music, frame-perfect

Creative Playground
Source Video Loader: Load ANALYSE Video

Simple Dance Poser: Pick built-in or library style, tweak parameters, animate

Library Building
Extract dances from any video for reuse and re-sync

Build up your own JSON dance library over time

🛠️ Technical Details
Pose format: 128 points, normalized (0.0–1.0), JSON with metadata

BPM detection: librosa-powered, 45–220 BPM range supported

Sync: frame-perfect, beat-aligned, time-stretch (drift-resistant)

Performance: CUDA-accelerated where available, efficient tensor code

💡 Best Practices
Use recommended resolutions (832×468 or 468×832)

Prefer clear, well-lit, full-body videos (minimize motion blur)

For accurate BPM, use clean music and clear audio tracks

Always review skeleton output for distortion (check video input ratio)

🔧 Troubleshooting
Missing model error:
Ensure both dw-ll_ucoco_384.onnx and yolox_l.onnx are in ComfyUI\models\dwpose\

Can't find dances:
Check both library folders above for your saved JSONs

Saving error:
Verify you have write permission to the custom_nodes\BAIS1Cs_VACE_DANCE_SYNC_SUITE\ folder

FFmpeg not found:
Download FFmpeg, add its bin folder to your Windows PATH

Distorted skeletons:
Only use supported resolutions/aspect ratios

📚 More Info
See bundled documentation for advanced use or submit issues on GitHub.

For further help, tag @BAIS1C on Discord or open an issue.

Enjoy rapid, pro-level dance animation and music sync with BAIS1C’s VACE Dance Sync Suite!

