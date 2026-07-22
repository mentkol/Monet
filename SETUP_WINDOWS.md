# Monet — Windows Setup Guide (Intel Core Ultra / Arc laptops)

Complete, step-by-step setup for a Windows 11 machine (e.g. Acer Swift Go 14,
Intel Core Ultra 5 125H / 16GB / Intel Arc iGPU).

This guide is split into two parts:

- **Part 1 — Get it working on CPU (guaranteed).** Do this fully first.
- **Part 2 — Enable Intel Arc GPU (XPU) for speed (optional).** Add this after Part 1 works.

---

## Part 1 — Set up the project (works on CPU)

Do all of this on the Windows machine.

### Step 1 — Install Git

1. Go to **https://git-scm.com/download/win** and download the **64-bit Git for Windows Setup**.
2. Run the installer. **Keep all the default options** — just click "Next" through every screen. This also installs Git Credential Manager (lets you log into GitHub via a browser popup).
3. Verify: open **Command Prompt** (press `Win`, type `cmd`, Enter) and run:
   ```
   git --version
   ```
   You should see something like `git version 2.45.x`.

### Step 2 — Install Python 3.11

1. Go to **https://www.python.org/downloads/release/python-3119/** (any 3.11.x is fine) and download **Windows installer (64-bit)**.
2. Run the installer. ⚠️ **CRITICAL: on the first screen, check the box "Add python.exe to PATH"** at the bottom before clicking "Install Now". This is the most common beginner mistake.
3. Verify in Command Prompt:
   ```
   python --version
   ```
   Should print `Python 3.11.x`.

### Step 3 — Download (clone) your project

1. Decide where you want the project, e.g. `C:\Users\<yourname>\Documents\Projects`.
2. Open **Command Prompt** and navigate there (create the folder if needed):
   ```
   mkdir C:\Users\%USERNAME%\Documents\Projects
   cd C:\Users\%USERNAME%\Documents\Projects
   ```
3. Clone your repo:
   ```
   git clone https://github.com/mentkol/Monet.git
   ```
   - If a browser/GitHub login window pops up, sign in with your GitHub account and authorize.
   - This creates a folder `Monet` with all the code.
4. Go into it:
   ```
   cd Monet
   ```

### Step 4 — Create a Python virtual environment

A virtual environment keeps this project's packages separate from everything else. **Always activate it before running the project.**

1. Still in Command Prompt, inside the `Monet` folder, run:
   ```
   python -m venv .venv
   ```
   (takes ~10 seconds — creates a `.venv` folder)
2. **Activate** it:
   ```
   .venv\Scripts\activate
   ```
   You'll now see `(.venv)` at the start of your prompt line. That means it's active. ✅
   - **PowerShell users:** if you get a red "running scripts is disabled" error, run this once, then retry activation:
     ```
     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
     ```
     (Type `A` + Enter if asked.)

### Step 5 — Install the dependencies

Run these **one at a time**, in order. The first installs a smaller CPU-only PyTorch (your Acer has no NVIDIA chip, so this saves ~2GB of unnecessary download):

```
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
```

Then the rest:

```
pip install -r server\requirements.txt
```

This downloads a lot (transformers, opencv, scikit-learn, etc.) — **5-10 minutes** depending on your connection. Let it finish. The torch lines will say "already satisfied" (good — it keeps the CPU build you just installed).

> If you see a long error ending in a wall of red text, copy the last ~10 lines — most often it's a network blip, fixed by re-running the same command.

### Step 6 — Run the server (first run downloads the AI models)

1. Move into the server folder:
   ```
   cd server
   ```
2. Start the server:
   ```
   python server.py
   ```
3. **First run only:** it will print `Downloading model weights (one-time)...` and download **~5-6GB** of models (the trained detector + the SigLIP/DINOv2 base models). This needs internet and takes a while. Leave it alone until it finishes.
4. You'll eventually see the startup banner ending with `Ready & Waiting for extension`. **The server is running.** Keep this window open.
   - The line `Device: cpu (float32) — warmup OK` confirms it's on CPU for now (we'll switch to Arc GPU in Part 2).

### Step 7 — Load the browser extension (Chrome or Edge)

1. Open Chrome (or Edge).
2. Go to: **`chrome://extensions`** (or `edge://extensions` in Edge).
3. Turn on **Developer mode** (top-right toggle).
4. Click **Load unpacked**.
5. Select the folder: `C:\Users\<yourname>\Documents\Projects\Monet\extension\chrome`
6. The extension is now installed. (Use the `extension\firefox` folder only if you use Firefox.)

### Step 8 — Test it

1. With the server running and the extension loaded, go to **YouTube Shorts** in that browser.
2. As a Short plays, the extension should detect it and the server's Command Prompt window should print a result line like `[Xms] ... (Score: 0.xx)` within a few seconds, and a colored badge appears on the video.

🎉 **You now have a fully working project.** Jump to **Day-to-day usage** below, or continue to Part 2 to make it much faster using the Arc GPU.

---

## Part 2 — Enable Intel Arc GPU (XPU) for big speedup [optional]

Only do this **after Part 1 works on CPU**. This replaces CPU inference with the Arc iGPU — should cut per-video time roughly in half or better. If anything breaks, an easy way back to CPU is included.

### Step X1 — Install Intel XPU PyTorch + IPEX

Open a Command Prompt, `cd` into your project, activate the venv:
```
cd C:\Users\%USERNAME%\Documents\Projects\Monet
.venv\Scripts\activate
```

Install the Arc-enabled PyTorch (large download, ~2GB):
```
pip install --upgrade --force-reinstall torch torchvision --index-url https://pytorch-extension.intel.com/release-whl/stable-xpu/us/
```

Then install Intel Extension for PyTorch (the thing that powers Arc):
```
pip install intel-extension-for-pytorch
```

### Step X2 — Verify the install

```
python -c "import torch, intel_extension_for_pytorch; print('xpu available:', torch.xpu.is_available()); print('device count:', torch.xpu.device_count())"
```
You want to see `xpu available: True`. If it says `False` or errors, your Acer's Arc drivers may need updating — get the latest **Intel Arc GPU driver** from Intel's site or your Acer updater, then retry.

### Step X3 — Run the server and confirm it uses XPU

```
cd server
python server.py
```
This time, look for the startup line:
```
Device: xpu (bfloat16) — warmup OK (batch=3)
```
✅ If you see `xpu`, the Arc GPU is now doing the heavy lifting. Test a Short and check the new `[Xms]` time.

### Troubleshooting Part 2 (if it falls back to CPU or errors)

- **If it prints `Device: cpu` instead of `xpu`**: IPEX/torch versions didn't match. This is the one fragile part. Try reinstalling IPEX to match, or just accept CPU (it still works).
- **If `ipex optimize: skipped (...)` prints**: disable it by running the server with:
  ```
  set MONET_IPEX_OPTIMIZE=0
  python server.py
  ```
  Plain XPU still gives most of the speedup.
- **Emergency revert to the clean CPU build** (guaranteed working, like Part 1):
  ```
  pip install --upgrade --force-reinstall torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
  pip uninstall -y intel-extension-for-pytorch
  ```

---

## Day-to-day usage (every time you want to use it)

1. Open Command Prompt, go to the project, activate the venv, run the server:
   ```
   cd C:\Users\%USERNAME%\Documents\Projects\Monet\server
   ..\.venv\Scripts\activate
   python server.py
   ```
2. Wait for `Ready & Waiting for extension`.
3. Browse YouTube Shorts — the extension (already loaded in your browser) talks to the server automatically.
4. To stop the server: click its Command Prompt window and press `Ctrl + C`.

## Getting code updates later

When you change code on another machine and push, pull it here:
```
cd C:\Users\%USERNAME%\Documents\Projects\Monet
git pull
```
(If dependencies changed, re-run `pip install -r server\requirements.txt` while the venv is active.)

---

## Environment variable tuning knobs

All optional. Set with `set VAR=value` in Command Prompt before `python server.py`.

| Variable | Default | Purpose |
|---|---|---|
| `MONET_DEVICE` | auto | Force `cpu` / `xpu` / `cuda` / `mps` |
| `MONET_DETECTOR_FRAMES` | `3` | Number of frames the AI detector scores (lower = faster) |
| `MONET_XPU_DTYPE` | `bf16` | XPU precision; try `fp16` |
| `MONET_IPEX_OPTIMIZE` | `1` | `0` to skip `ipex.optimize()` |
| `MONET_TORCH_THREADS` | cpu_count − 3 | PyTorch CPU threads |
| `MONET_BATCH_SIZE` | `4` | Detector batch size |
| `MONET_OCR` | `0` | `1` to enable Tesseract OCR (requires the Tesseract engine installed) |

## Notes

- **Disk:** reserve ~8GB free (models + packages).
- **RAM:** 16GB is plenty — the model uses ~3GB.
- **First video after starting** the server may be slightly slow (GPU warms up); later videos are fast.
- The `.venv` folder and `server/models/` are **not** in git — each machine builds its own.
