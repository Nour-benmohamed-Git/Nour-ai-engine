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

## Step 3 — Upload your server file to /workspace

Use RunPod's file upload UI or run:

```bash
# if you paste the file via the terminal
nano /workspace/asr_server.py
# paste your code, Ctrl+X to save
```

## Step 4 — Run the server

```bash
cd /workspace
python asr_server.py
```
