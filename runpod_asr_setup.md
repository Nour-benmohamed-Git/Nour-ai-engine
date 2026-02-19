# ASR Server Setup Steps

**12:16 AM**

You're in the wrong folder. Navigate to your Desktop first:

```powershell
cd $env:USERPROFILE\OneDrive\Desktop
```

Then run:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("$pwd\asr_server.py"))
```

## Step 1 — Install system audio deps

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg sox libsox-fmt-all
```

## Step 2 — Install NeMo + server deps

```bash
pip install "nemo_toolkit[asr]>=1.22.0" fastapi "uvicorn[standard]" --extra-index-url https://pypi.org/simple
```

## Step 3 — Fix torchvision/torch CUDA mismatch

RunPod images often ship with a mismatched `torchvision` build (different CUDA tag than `torch`), which causes this crash at import time:

```
RuntimeError: operator torchvision::nms does not exist
```

First, check your exact versions:

```bash
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"
pip show torchvision | grep Version
```

Then reinstall `torchvision` from the index that matches your torch CUDA build. Use the **nightly** index if torch is a nightly/pre-release build (e.g. `2.10.0+cu128`):

```bash
# For nightly torch builds (e.g. 2.10.0+cu128)
pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# For stable torch builds — replace cu128 with your actual CUDA tag
pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify both now share the same CUDA tag:

```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)"
```

Both should show the same `cuXXX` suffix (e.g. `cu128`).

## Step 4 — Upload your server file to /workspace

Use RunPod's file upload UI or run:

```bash
# if you paste the file via the terminal
nano /workspace/asr_server.py
# paste your code, Ctrl+X to save
```

## Step 5 — Run the server

```bash
cd /workspace
python asr_server.py
```