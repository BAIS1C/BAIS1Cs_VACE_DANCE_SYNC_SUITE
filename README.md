🕺 BAIS1C VACE Dance Sync Suite (BETA STILKLL BROKEN BUT TESTING)
Professional toolkit for pose extraction, library management, and music-driven dance animation in ComfyUI.
Full pipeline: video-to-pose extraction, library export, music-reactive dance generation, and precise BPM synchronization.

🚀 Installation
1. Clone the Repo

bash
Copy
Edit
cd custom_nodes
git clone https://github.com/BAIS1C/BAIS1Cs_VACE_DANCE_SYNC_SUITE.git
2. Install Python Dependencies

bash
Copy
Edit
pip install -r custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/requirements.txt
3. Download Required DWPose Models

dw-ll_ucoco_384.onnx

yolox_l.onnx

Place models in:

bash
Copy
Edit
ComfyUI/models/dwpose/
  ├── dw-ll_ucoco_384.onnx
  └── yolox_l.onnx
(Or set a custom path via DWPose environment variable.)

📋 Video Format & Pose Model Requirements
Aspect ratios: 16:9 (landscape) or 9:16 (portrait)

Recommended resolutions:

Landscape: 832×468

Portrait: 468×832

Non-standard aspect ratios will distort pose detection and skeleton output.

🎭 Available Nodes
1. 🎥 BAIS1C Source Video Loader
Category: BAIS1C VACE Suite/Source

Loads video, extracts audio, BPM, frame rate, and all sync metadata.

Outputs: processed video, audio, fps, bpm, frame count, duration, sync_meta dict.

2. 🎯 BAIS1C Pose Extractor (128pt)
Category: BAIS1C VACE Suite/Extraction

Extracts 128-point pose tensors from video (DWPose), auto-saves as JSON for reuse.

Format: 23 body, 68 face, 21+16 hand keypoints (normalized).

Inputs: video, sync_meta, title, debug, etc.

Outputs: pose tensor (TENSOR), sync_meta (DICT).

3. 🎵 BAIS1C Music Control Net (Auto-Sync)
Category: BAIS1C VACE Suite/Control

Professional BPM-to-pose sync; frame-perfect timing, various sync/loop modes.

Inputs: audio, target_fps, pose source (tensor or library), sync/loop methods, etc.

Outputs: synced pose tensor, video visualization, sync report.

4. 🕺 BAIS1C Simple Dance Poser (Creative Playground)
Category: BAIS1C VACE Suite/Creative

Fast, creative dance animation from library or built-in styles; user-tweakable speed, smoothing, music response.

Inputs: audio, dance source, style, speed, reactivity, smoothing, etc.

Outputs: animated pose tensor, video, info report.

📂 File Structure
Model files: ComfyUI/models/dwpose/

Dance library:

Main: custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/dance_library/

Fallback: ComfyUI/output/dance_library/

Starter dances:

basic_walking.json

arms_up_body_sway.json

hands_on_hips.json

🔄 Example Workflows COMING SOON
Professional Sync
🎥 Source Video Loader → 2. 🎯 Pose Extractor → 3. 🎵 Music Control Net

Creative Playground
🎥 Source Video Loader → 2. 🕺 Simple Dance Poser

Library Building
Use Pose Extractor to build JSON library from any source

Sync/re-mix with Music Control Net or Simple Dance Poser

🛠️ Technical Details
Pose format: 128 points, normalized [0.0–1.0], JSON with metadata

BPM detection: librosa, validated, supports 45–220 BPM

Sync: frame-perfect, beat-aligned, time-stretch; drift-resistant

Performance: CUDA-accelerated where available, efficient tensor logic

💡 Tips
Use recommended resolutions for best results

Prefer clear, full-body, well-lit videos

For accurate BPM: use steady, high-quality music

Always check output video for skeleton distortion (fix input aspect ratio if needed)

🔧 Troubleshooting
Missing models: Double-check model ONNX files are in ComfyUI/models/dwpose/

No dances saved: Check both library folders above

Saving errors: Fix permissions on custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/

Distorted skeletons: Use only 832×468 or 468×832 inputs

📚 More Info
Bundled docs and examples included

For bugs, open a GitHub issue

Enjoy next-level dance motion and music sync with BAIS1C’s VACE Dance Sync Suite.

