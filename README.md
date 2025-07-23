# 🕺 BAIS1C's VACE Dance Sync Suite

**Professional toolkit for pose extraction, library management, and music-driven dance animation in ComfyUI.**

A complete dance synchronization pipeline featuring pose tensor extraction, music-reactive animation, and professional BPM synchronization algorithms.

---

## 🚀 Installation

### 1. Clone the Repository

From your ComfyUI root folder:

```bash
cd custom_nodes
git clone https://github.com/BAIS1C/BAIS1Cs_VACE_DANCE_SYNC_SUITE.git
```

### 2. Install Requirements
After cloning, install dependencies:

```bash
pip install -r custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/requirements.txt
```

### 3. Download and Place DWPose Models
Required model files:
- `dw-ll_ucoco_384.onnx`
- `yolox_l.onnx`

Create the following folder (if it doesn't exist):

```bash
ComfyUI/models/dwpose/
```

Place both .onnx files inside `ComfyUI/models/dwpose/`, so you have:

```
ComfyUI/
└── models/
    └── dwpose/
        ├── dw-ll_ucoco_384.onnx
        └── yolox_l.onnx
```

**Note:** You may use a different folder by setting the `DWPose` environment variable, but `ComfyUI/models/dwpose/` is the default and recommended path.

---

## 📋 **IMPORTANT: Video Format Requirements**

**For optimal pose detection and skeleton rendering:**

### **Recommended Aspect Ratios:**
- **16:9 (Landscape)** - Standard widescreen format
- **9:16 (Portrait)** - Mobile/vertical content format

### **Recommended Resolutions:**
- **832×468** (16:9 landscape) - Ideal for most dance content
- **468×832** (9:16 portrait) - Perfect for mobile/vertical videos

### **Why These Ratios Matter:**
- Prevents skeleton distortion during pose detection
- Ensures accurate body proportion mapping
- Maximizes DWPose model compatibility
- Maintains professional video standards

**⚠️ Using non-standard aspect ratios may result in distorted pose detection and skeleton rendering.**

---

## 🎭 Available Nodes

### **🎥 BAIS1C Source Video Loader**
*Category: BAIS1C VACE Suite/Source*

**Purpose:** Loads video files and extracts comprehensive metadata for the dance sync pipeline.

**Features:**
- Enhanced BPM detection with cross-validation (45-220 BPM range)
- Full audio duration analysis using librosa
- Robust FFmpeg audio extraction with error handling
- Multiple fallback BPM detection methods
- Automatic aspect ratio validation and guidance

**Inputs:**
- `video` (VIDEO) - Source video file

**Outputs:**
- `video_obj` (VIDEO) - Processed video object
- `audio_obj` (AUDIO) - Extracted audio with metadata
- `fps` (FLOAT) - Video frame rate
- `bpm` (FLOAT) - Detected audio BPM
- `frame_count` (INT) - Total video frames
- `duration` (FLOAT) - Video duration in seconds

### **🎯 BAIS1C Pose Extractor (128pt)**
*Category: BAIS1C VACE Suite/Extraction*

**Purpose:** Extracts 128-point pose tensors from video using DWPose detection and saves them as JSON files for reuse.

**Features:**
- DWPose-based pose detection with 128-point format
- Comprehensive pose format: 23 body + 68 face + 42 hand keypoints
- Automatic JSON library management
- Progress tracking with tqdm
- Robust error handling and fallback poses
- Metadata preservation (BPM, FPS, duration)

**128-Point Format Structure:**
- **Body keypoints:** 0-22 (23 points)
- **Face keypoints:** 23-90 (68 points)  
- **Left hand:** 91-111 (21 points)
- **Right hand:** 112-127 (16 points)

**Inputs:**
- `video` (VIDEO) - Source video
- `audio` (AUDIO) - Audio metadata
- `fps` (FLOAT) - Video frame rate
- `bpm` (FLOAT) - Audio BPM
- `frame_count` (INT) - Total frames
- `duration` (FLOAT) - Video duration
- `sample_stride` (INT) - Frame sampling rate (1-10)
- `title` (STRING) - Output filename
- Optional: `author`, `style`, `tempo`, `description`, `debug`

**Outputs:**
- `metadata_summary` (STRING) - Extraction report
- `pose_tensor` (POSE) - 128-point pose tensor

### **🎵 BAIS1C Music Control Net** *(The Magic Sync Node)*
*Category: BAIS1C VACE Suite/Control*

**Purpose:** Professional music-to-pose synchronization using industry-standard BPM/FPS formulas. Loads existing pose JSONs and syncs them to new music with frame-perfect accuracy.

**Features:**
- Professional BPM synchronization algorithms
- Multiple sync methods (time_domain, frame_perfect, beat_aligned)
- Automatic pose library loading
- Frame-perfect timing to prevent drift
- Smart looping and cropping modes
- High-quality pose visualization
- Synced JSON export capability

**Sync Methods:**
- **Time Domain:** Simple time stretching (your original approach)
- **Frame Perfect:** Professional frame-perfect sync to prevent drift
- **Beat Aligned:** Ensures cuts happen exactly on beats

**Inputs:**
- `audio` (AUDIO) - Target audio
- `target_bmp` (FLOAT) - Desired BPM (30-300)
- `target_fps` (FLOAT) - Output frame rate (12-60)
- `pose_source` - Library JSON or direct tensor
- `library_dance` - Available dance selections
- `sync_method` - Synchronization algorithm
- `loop_mode` - Fitting behavior
- `generate_video` (BOOLEAN) - Create visualization
- `video_style` - Stickman, dots, or skeleton
- `save_synced_json` (BOOLEAN) - Export synced results
- Output resolution: `width` (832) × `height` (468)
- Optional: `input_pose_tensor` (POSE)

**Outputs:**
- `synced_pose_tensor` (POSE) - Time-synchronized poses
- `pose_video` (IMAGE) - Visualization sequence
- `sync_report` (STRING) - Detailed synchronization report

### **🕺 BAIS1C Simple Dance Poser** *(Creative Playground)*
*Category: BAIS1C VACE Suite/Creative*

**Purpose:** Creative experimentation node for generating animated dance sequences with user-friendly controls for speed, smoothing, and music reactivity.

**Features:**
- Built-in dance styles (hip-hop, ballet, freestyle, bounce, robot)
- Real-time speed adjustment (0.1x - 3.0x)
- Movement smoothing controls
- Basic music reactivity (beat, bass, energy)
- Multiple visualization styles
- No complex sync requirements - just creative fun!

**Built-in Dance Styles:**
- **Hip-hop:** Dynamic arm movements with hip action
- **Ballet:** Graceful positions with elegant flow
- **Freestyle:** Free-flowing multi-directional movement
- **Bounce:** Rhythmic bouncing motion
- **Robot:** Discrete robotic pose positions

**Inputs:**
- `audio` (AUDIO) - Music source
- `dance_source` - Library or built-in styles
- `library_dance` - Available library dances
- `built_in_style` - Pre-made dance styles
- `dance_speed` (FLOAT) - Speed multiplier (0.1-3.0x)
- `movement_smoothing` (FLOAT) - Motion smoothing (0.0-0.9)
- `music_reactivity` (FLOAT) - Music response strength
- `react_to` - Beat, bass, energy, or none
- `visualization` - Stickman, dots, skeleton, or none
- Output resolution: `width` (832) × `height` (468)

**Outputs:**
- `animated_poses` (POSE) - Generated animation sequence
- `dance_video` (IMAGE) - Visualization
- `creation_info` (STRING) - Animation details

---

## 📂 File Structure & Storage

### **Pose Model Files:**
Located in `ComfyUI/models/dwpose/` (see installation above)

### **Dance Library (.json):**
Your extracted and saved dances appear in:
- **Primary:** `custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/dance_library/` (default)
- **Fallback:** `ComfyUI/output/dance_library/` (if primary not writable)

### **Included Starter Dances:**
The suite includes these default 128-point dance files:
- `basic_walking.json` - Natural walking motion with arm swing
- `arms_up_body_sway.json` - Rhythmic body sway with raised arms  
- `hands_on_hips.json` - Confident stance with weight shifting

---

## 🔄 Complete Workflows

### **Professional Sync Pipeline:**
1. **🎥 Source Video Loader** → Load video + extract BPM
2. **🎯 Pose Extractor** → Extract poses + save JSON with metadata  
3. **🎵 Music Control Net** → Load JSON + sync to new music → Perfect sync!

### **Creative Experimentation Pipeline:**
1. **🎥 Source Video Loader** → Load audio
2. **🕺 Simple Dance Poser** → Pick style + tweak parameters → Creative animation!

### **Library Building Workflow:**
1. Use **Pose Extractor** to build your dance library from video sources
2. Use **Music Control Net** to sync any library dance to new music
3. Use **Simple Dance Poser** for quick creative variations

---

## 🛠️ Technical Specifications

### **Pose Format:**
- **128-point comprehensive format**
- Normalized coordinates (0.0-1.0)
- JSON structure with full metadata
- Compatible with DWPose detection pipeline

### **Audio Analysis:**
- Full-duration BPM detection using librosa
- Cross-validation with multiple algorithms
- Support for 45-220 BPM range (all musical genres)
- Beat detection and frequency analysis

### **Synchronization:**
- Professional BPM/FPS mathematical formulas
- Frame-perfect timing algorithms
- Multiple sync methods for different use cases
- Drift prevention for long sequences

### **Performance:**
- CUDA acceleration where available
- Efficient pose tensor operations
- Memory-optimized video processing
- Progress tracking for long operations

---

## 💡 Tips & Best Practices

### **For Best Pose Detection:**
- Use videos with clear, well-lit subjects
- Ensure full body is visible in frame
- Avoid heavy motion blur or compression artifacts
- **Use 832×468 or 468×832 resolutions for optimal results**

### **For Music Synchronization:**
- Extract poses from videos with clear, consistent BPM
- Use high-quality audio for accurate BPM detection
- Choose appropriate sync method based on use case
- Test with different loop modes for best fit

### **For Creative Animation:**
- Start with built-in styles and modify parameters
- Use music reactivity for dynamic movement
- Experiment with different speed multipliers
- Combine library dances with creative controls

---

## 🔧 Troubleshooting

### **Missing model error?**
Double-check that `dw-ll_ucoco_384.onnx` and `yolox_l.onnx` are in `ComfyUI/models/dwpose/`

### **Can't find your dances?**
Look in both `dance_library/` folders listed above.

### **Saving error?**
Ensure you have write permissions to the `custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/` folder.

### **FFmpeg not found?**
Install FFmpeg and ensure it's in your system PATH for audio extraction.

### **Distorted skeletons?**
Check your video aspect ratio - use 16:9 (832×468) or 9:16 (468×832) formats.

---

## 📚 More Info

For additional help, see bundled documentation or open an issue on GitHub.

**Enjoy creating and remixing dance motion with music using BAIS1C's VACE Dance Sync Suite!**

---

*This suite represents cutting-edge pose-to-music synchronization technology, combining professional BPM analysis, advanced pose detection, and frame-perfect timing algorithms for sector-leading dance animation capabilities.*