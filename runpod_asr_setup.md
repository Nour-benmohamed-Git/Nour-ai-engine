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

## Step 3 — Fix torch/torchvision/torchaudio CUDA mismatch

RunPod images often ship with mismatched builds of `torchvision` and `torchaudio` (different CUDA tag than `torch`), causing these crashes at import time:

```
RuntimeError: operator torchvision::nms does not exist
OSError: libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
```

First, check your exact versions:

```bash
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"
pip show torchvision torchaudio | grep -E "^(Name|Version)"
```

Then reinstall all three from the matching index. Use the **nightly** index if torch is a nightly/pre-release build (e.g. `2.10.0+cu128`). Note `--force-reinstall` is required — pip will incorrectly report packages as already satisfied without it:

```bash
# For nightly torch builds (e.g. 2.10.0+cu128)
pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# For stable torch builds — replace cu128 with your actual CUDA tag
pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **Why `--no-deps` for torchaudio?** Without it, pip also upgrades torch to the latest nightly, which then breaks NeMo's fsspec and other dependencies. `--no-deps` installs only torchaudio itself.

Verify all three share the same CUDA tag:

```bash
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)"
```

All three should show the same `cuXXX` suffix (e.g. `2.12.0.dev+cu128 0.26.0.dev+cu128 2.11.0.dev+cu128`).

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