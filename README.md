# 🎭 BAIS1C VACE Dance Sync Suite

<div align="center">

**Modular ComfyUI Node Collection for Procedural, Audio-Reactive, and Filmmaking-Grade Pose & Dance Animation**

![GitHub stars](https://img.shields.io/github/stars/yourusername/bais1c-vace-dance-sync?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/bais1c-vace-dance-sync?style=social)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-green.svg)

</div>

---

## 🚦 Workflow Overview

The pipeline is designed for **fast, robust, and modular pose/music extraction and sync**:
1. **Load your source video** (or images/audio) using a ComfyUI video/image node (e.g. `VHS_LoadVideo`).
2. **Feed extracted images and audio** (plus video_info/meta) to the **BAIS1C_SourceVideoLoader** node.
3. **BAIS1C_SourceVideoLoader**:
    - Passes through images and audio.
    - *Augments* sync metadata (BPM, duration, FPS, etc) using **robust multi-method audio analysis**.
    - Outputs unified `sync_meta` dict, plus UI info string for diagnostics.
4. **BAIS1C_PoseExtractor**:
    - Uses only images and the `sync_meta`.
    - Performs framewise pose estimation (DWPose) and attaches all relevant metadata.
5. **Music Control, Dance Poser, Save to JSON**:
    - Downstream nodes (e.g. `BAIS1C_MusicControlNet`, `BAIS1C_SimpleDancePoser`, etc) work directly from the unified meta, images, and audio, with zero manual BPM/FPS input needed.
    - Each node passes the `sync_meta` forward, so pipeline context is always maintained.
6. **Final video or JSON is ready for VACE, WAN, or further remix.**

---

## 🛠️ Node Collection (Current & Planned)

| Node                         | Purpose                         | Key Features                                         |
|------------------------------|---------------------------------|------------------------------------------------------|
| **📦 BAIS1C_SourceVideoLoader** | *NEW: Minimal loader/meta prep* | Audio/video/image pass-through, robust BPM, meta UI  |
| **🎬 BAIS1C_PoseExtractor**     | Pose extraction                | DWPose, batch, meta in/out, no FPS/BPM manual input  |
| **🎵 BAIS1C_MusicControlNet**   | Pose/music sync, modulation    | Enhanced BPM, beat/energy mapping, meta-driven sync  |
| **💃 BAIS1C_SimpleDancePoser**  | Parametric dance generation    | Minimal, meta-driven, supports music reactivity      |
| **💾 BAIS1C_SavePoseJSON**      | Export pose data               | Metadata-locked JSON, ready for library/VACE/WAN     |

---

## 🧠 New Pipeline Example

```mermaid
graph TD
    A[🎦 VHS_LoadVideo/Images+Audio] --> B[📦 SourceVideoLoader (BPM, meta, UI)]
    B --> C[🎬 PoseExtractor (images, meta)]
    C --> D[🎵 MusicControlNet (images, meta, audio)]
    D --> E[💃 SimpleDancePoser (poses, meta, audio)]
    E --> F[💾 SavePoseJSON (library, meta)]
🔥 Key Changes vs Old Pipeline
No manual FPS or BPM entry: All nodes receive these via sync_meta.

Audio may be omitted: If not required, audio is simply ignored downstream.

Upstream nodes supply all meta: e.g. from VHS, VideoHelperSuite, etc.

Downstream: always pass sync_meta. No re-entry of technical parameters.

UI info string: At each step, you see a summary (frames, FPS, BPM, duration, etc).

🚀 Installation & Setup
Clone this repo:

bash
Copy
Edit
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/yourusername/bais1c-vace-dance-sync.git
Install Dependencies:

bash
Copy
Edit
cd bais1c-vace-dance-sync
pip install -r requirements.txt
Place DWPose models in /models/dwpose/:

yolox_l.onnx

dw-ll_ucoco_384.onnx

Restart ComfyUI.

(Optional) Create dance_library/ for saving pose JSON.

📋 Requirements
Add this to your requirements.txt:

txt
Copy
Edit
torch
numpy
librosa
opencv-python
onnxruntime
decord
scipy             # <--- needed for improved BPM/analysis code
If you want to make the new enhanced_audio_analysis.py a separate module, ensure it’s in the repo root or /nodes/ and import as from .enhanced_audio_analysis import EnhancedAudioAnalyzer.

📝 Example Usage
Video/Audio Load: Use VHS_LoadVideo, VideoHelperSuite, or any node that outputs images, audio, and video_info (dict/meta).

SourceVideoLoader: Receives these, calculates BPM, passes through all data, outputs sync_meta dict.

PoseExtractor: Reads images, sync_meta only; outputs pose tensor + meta.

(Optional) Music ControlNet: Reads pose, audio, meta; applies beat/energy modulation as desired.

(Optional) SavePoseJSON: Exports pose+meta to JSON for library/VACE/WAN use.

💡 Notes
Pass sync_meta forward at every step!

All nodes now share a unified metadata flow.

Audio is optional; pass None if not needed.

BPM/FPS is always auto-detected—no manual entry required.

🗺️ Roadmap and Integration
Cinematic & VACE workflows: All outputs are fully compatible with VACE/WAN (see Knowledge Base)

Further node expansion: Planned for camera motion tagging, action category/labeling, and deep remix support.

Library-first approach: Everything in /dance_library/ is portable/remixable for future batch generation.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

<div align="center">
Made with ❤️ by the BAIS1C Team

Every saved pose or action is fully remixable, and ready for next-gen AI video, VACE, or experimental filmmaking.

⭐ Star this repo if you found it helpful! ⭐