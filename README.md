# 🕺 BAIS1C's VACE Dance Sync Suite

**Toolkit for pose extraction, library management, and music-driven dance animation in ComfyUI.**

---

## 🚀 Installation

### 1. Clone the Repository

From your ComfyUI root folder:

```bash
cd custom_nodes
git clone https://github.com/BAIS1C/BAIS1Cs_VACE_DANCE_SYNC_SUITE.git
2. Install Requirements
After cloning, install dependencies:

bash
Copy
Edit
pip install -r custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/requirements.txt
3. Download and Place DWPose Models
Required model files:

dw-ll_ucoco_384.onnx

yolox_l.onnx

Create the following folder (if it doesn't exist):

bash
Copy
Edit
ComfyUI/models/dwpose/
Place both .onnx files inside ComfyUI/models/dwpose/, so you have:

markdown
Copy
Edit
ComfyUI/
└── models/
    └── dwpose/
        ├── dw-ll_ucoco_384.onnx
        └── yolox_l.onnx
Note:
You may use a different folder by setting the DWPose environment variable, but ComfyUI/models/dwpose/ is the default and recommended path.

📂 Where Are Files Stored?
Pose Model Files:

Go in ComfyUI/models/dwpose/ (see above).

Dance Library (.json):

Your extracted and saved dances will appear in:

custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/dance_library/ (default)

If that's not writable, fallback: ComfyUI/output/dance_library/

Bundled starter dances are shipped in the repo's dance_library/ folder.

🛠️ First Run
Restart ComfyUI after installation and model download.

The suite’s nodes will appear and operate automatically.

Extract poses from videos, sync to music, and browse or add new dances to your library!

💡 Troubleshooting
Missing model error?

Double-check that dw-ll_ucoco_384.onnx and yolox_l.onnx are in ComfyUI/models/dwpose/

Can't find your dances?

Look in both the dance_library/ folders above.

Saving error?

Ensure you have write permissions to the custom_nodes/BAIS1Cs_VACE_DANCE_SYNC_SUITE/ folder.

📚 More Info
For additional help, see bundled ReadMe.md or open an issue on GitHub.

Enjoy creating and remixing dance motion with music using BAIS1C's Dance Sync Suite!


