# 🎬 Japanese → English Subtitle Generator

A fully **offline**, GPU-accelerated tool that automatically transcribes Japanese videos and generates bilingual subtitles (Japanese + English) in `.srt` format.

---

## ✨ Features

- 🎥 Supports `.mp4` and `.mkv` video files
- ⚡ GPU-accelerated transcription via **faster-whisper** (NVIDIA CUDA)
- 🇯🇵→🇬🇧 Translates Japanese to English using **Helsinki-NLP MarianMT**
- 📄 Outputs bilingual `.srt` subtitles (Japanese on top, English below)
- 📁 Batch processing — single file, multiple files, or entire folders
- 🔒 Fully offline after first model download

---

## 🖥️ Verified Working Environment

| Component | Version |
|---|---|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10 or 3.11 |
| NVIDIA Driver | 572.61 |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 3080 Ti (12 GB VRAM) |

> ⚠️ **PyTorch must be ≥ 2.6** for MarianMT to load correctly.  
> ⚠️ **CUDA ≥ 12.3** is required for faster-whisper / ctranslate2 ≥ 4.5.0.

---

## 📦 Installation

> **Important:** Install PyTorch **before** faster-whisper. Wrong order causes PyTorch to be replaced with a CPU-only version.

### Step 1 — Uninstall existing PyTorch (clean slate)
```bash
pip uninstall torch torchvision torchaudio -y
```

### Step 2 — Install PyTorch with CUDA 12.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> For other CUDA versions:
> - CUDA 12.6 → `https://download.pytorch.org/whl/cu126`
> - CUDA 12.1 → `https://download.pytorch.org/whl/cu121`
> - CUDA 11.8 → `https://download.pytorch.org/whl/cu118`

### Step 3 — Install remaining dependencies
```bash
pip install faster-whisper transformers sentencepiece sacremoses
```

### Step 4 — Verify installation
```bash
python -c "
import torch
from faster_whisper import WhisperModel
from transformers import MarianMTModel
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('All good!')
"
```

Expected output:
```
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 3080 Ti
All good!
```

---

## ▶️ Usage

```bash
# Single file
python japanese_subtitles.py video.mkv
python japanese_subtitles.py video.mp4

# Multiple files
python japanese_subtitles.py ep1.mkv ep2.mkv ep3.mkv

# Entire folder
python japanese_subtitles.py C:\Videos\MyShow\
```

The `.srt` file is saved next to the video with the same filename.

---

## ⚙️ Configuration

Edit these variables at the top of `japanese_subtitles.py`:

```python
WHISPER_MODEL   = "large-v3"   # tiny / base / small / medium / large-v3
DEVICE          = "cuda"       # "cuda" for GPU, "cpu" for CPU only
COMPUTE_TYPE    = "float16"    # "float16" (GPU) or "int8" (CPU)
TRANSLATE_MODE  = "marianmt"   # "marianmt" (better) or "whisper" (faster, 1-step)
BILINGUAL       = True         # True = Japanese + English, False = English only
OUTPUT_FORMAT   = "srt"        # "srt" or "sbv"
```

### Translation Modes

| Mode | How | Quality | Speed |
|---|---|---|---|
| `marianmt` | Whisper transcribes → MarianMT translates | Better ⭐ | Slower (2 steps) |
| `whisper` | Whisper transcribes + translates in one step | Good | Faster (1 step) |

---

## 🎮 Whisper Model Guide

| Model | VRAM | Speed | Quality |
|---|---|---|---|
| `tiny` | ~1 GB | ~60x realtime | Basic |
| `small` | ~2 GB | ~30x realtime | Good |
| `medium` | ~5 GB | ~15x realtime | Very Good |
| `large-v3` | ~10 GB | ~6x realtime | Best ⭐ |

> With an RTX 3080 Ti (12 GB), `large-v3` is recommended for best accuracy.

---

## 📥 First Run — Model Downloads

On first run, these models are downloaded automatically (internet required once):

| Model | Size | Purpose |
|---|---|---|
| Whisper `large-v3` | ~3 GB | Japanese transcription |
| Helsinki-NLP `opus-mt-ja-en` | ~300 MB | Japanese → English translation |

After the first run, everything works **completely offline**. ✅

---

## 🔧 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `CUDA not available` | Wrong PyTorch build | Reinstall with correct `--index-url` |
| `MarianMT requires PyTorch >= 2.6` | Old PyTorch version | `pip install --upgrade torch` with cu128 URL |
| `CVE-2025-32434 ValueError` | torch.load security block | Already fixed in script with `use_safetensors=True` |
| `faster-whisper uninstalled PyTorch` | Wrong install order | Always install PyTorch FIRST |
| Symlinks warning on Windows | Windows Developer Mode off | Settings → System → For Developers → On (or ignore) |
| Out of VRAM | Model too large | Switch `WHISPER_MODEL` to `medium` or `small` |
| `n_gpu_layers` / `device` AttributeError | Using pywhispercpp (wrong lib) | Use `faster-whisper` instead |

---

## 📁 Repository Structure

```
├── japanese_subtitles.py   # Main script
├── README.md               # This file
└── LICENSE                 # MIT License
```

---

## 🤝 Dependencies

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Fast Whisper inference with CTranslate2
- [Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en) — Japanese→English MarianMT model
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — NLP model library

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
