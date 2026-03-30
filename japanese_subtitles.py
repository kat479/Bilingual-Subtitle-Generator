"""
Japanese → English Subtitle Generator
======================================
- Supports .mp4 and .mkv input
- GPU accelerated (NVIDIA CUDA) via faster-whisper
- Transcribes Japanese with faster-whisper (large-v3)
- Translates to English with MarianMT (Helsinki-NLP)
- Outputs bilingual .srt file (Japanese + English)

Install dependencies:
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install faster-whisper transformers sentencepiece sacremoses

Usage:
    python japanese_subtitles.py video.mp4
    python japanese_subtitles.py video.mkv
    python japanese_subtitles.py video1.mp4 video2.mkv   # batch
    python japanese_subtitles.py C:\\Videos\\             # entire folder
"""

# ── Suppress all warnings before any imports ────────────────────────────────────
import os
import sys
import warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "large-v3"   # tiny / base / small / medium / large-v3
DEVICE          = "cuda"       # "cuda" for NVIDIA GPU, "cpu" for CPU only
COMPUTE_TYPE    = "float16"    # "float16" (GPU) or "int8" (CPU)
TRANSLATE_MODE  = "marianmt"   # "marianmt" (better quality) or "whisper" (faster, 1-step)
BILINGUAL       = True         # True = Japanese + English, False = English only
OUTPUT_FORMAT   = "srt"        # "srt" or "sbv"
# ───────────────────────────────────────────────────────────────────────────────


# ── Dependency & version check ───────────────────────────────────────────────────
def check_dependencies():
    # Check PyTorch
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not found. Run:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # Check PyTorch version (MarianMT needs >= 2.6)
    major, minor = map(int, torch.__version__.split(".")[:2])
    if major < 2 or (major == 2 and minor < 6):
        print(f"❌ PyTorch {torch.__version__} is too old. MarianMT requires >= 2.6. Run:")
        print("   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # Check CUDA
    device = DEVICE
    compute_type = COMPUTE_TYPE
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available — falling back to CPU.")
        device = "cpu"
        compute_type = "int8"
    elif device == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}  |  PyTorch: {torch.__version__}")

    # Check faster-whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster-whisper not found. Run:  pip install faster-whisper")
        sys.exit(1)

    # Check transformers (for MarianMT)
    if TRANSLATE_MODE == "marianmt":
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            print("❌ transformers not found. Run:  pip install transformers sentencepiece sacremoses")
            sys.exit(1)

    return device, compute_type


# ── Translation (MarianMT) ────────────────────────────────────────────────────────
_translator = None

def load_marianmt(device):
    global _translator
    if _translator is None:
        from transformers import MarianMTModel, MarianTokenizer
        print("📥 Loading MarianMT translation model (downloads ~300MB on first run)...")
        model_name = "Helsinki-NLP/opus-mt-ja-en"
        tokenizer  = MarianTokenizer.from_pretrained(model_name)
        # use_safetensors=True avoids the torch.load CVE-2025-32434 vulnerability error
        model = MarianMTModel.from_pretrained(model_name, use_safetensors=True).to(device)
        _translator = (tokenizer, model, device)
        print("✅ Translation model ready.\n")
    return _translator

def translate_marianmt(text, device):
    import torch
    tokenizer, model, dev = load_marianmt(device)
    tokens = tokenizer(
        [text], return_tensors="pt",
        padding=True, truncation=True, max_length=512
    )
    tokens = {k: v.to(dev) for k, v in tokens.items()}
    with torch.no_grad():
        translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


# ── Subtitle time formatting ──────────────────────────────────────────────────────
def srt_time(seconds):
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def sbv_time(seconds):
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h}:{m:02}:{s:02}.{ms:03}"


# ── Write subtitle file ───────────────────────────────────────────────────────────
def write_subtitles(segments, output_path, device, use_marianmt=False):
    print(f"📝 Writing subtitles → {os.path.basename(output_path)}")
    is_srt = output_path.endswith(".srt")
    total  = len(segments)
    idx    = 1

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            ja_text = seg.text.strip()
            if not ja_text:
                continue

            if use_marianmt:
                en_text = translate_marianmt(ja_text, device)
                print(f"  [{idx}/{total}] {ja_text[:40]}")
                print(f"           → {en_text[:40]}")
            else:
                en_text = ja_text  # whisper task=translate already outputs English

            # Bilingual: Japanese on top, English below
            text = f"{ja_text}\n{en_text}" if (BILINGUAL and use_marianmt) else en_text

            if is_srt:
                f.write(f"{idx}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{text}\n\n")
            else:
                f.write(f"{sbv_time(seg.start)},{sbv_time(seg.end)}\n{text}\n\n")

            idx += 1

    print(f"\n✅ {idx - 1} subtitle entries written → {output_path}")


# ── Process a single video ────────────────────────────────────────────────────────
def process_video(video_path, device, compute_type):
    from faster_whisper import WhisperModel

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in (".mp4", ".mkv"):
        print(f"⚠️  Skipping unsupported format: {video_path}")
        return

    base     = os.path.splitext(video_path)[0]
    out_path = f"{base}.{OUTPUT_FORMAT}"

    print(f"\n{'─' * 60}")
    print(f"🎬 Processing: {os.path.basename(video_path)}")
    print(f"{'─' * 60}")

    # 1. Load Whisper
    print(f"🤖 Loading Whisper [{WHISPER_MODEL}] on {device.upper()} ({compute_type})...")
    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)

    # 2. Transcribe (or translate in one step if TRANSLATE_MODE == "whisper")
    print("🔊 Transcribing...")
    task = "translate" if TRANSLATE_MODE == "whisper" else "transcribe"
    segments, info = model.transcribe(
        video_path,
        language="ja",
        task=task,
        beam_size=5,
        vad_filter=True,            # skip silent sections for speed
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    segments = list(segments)
    print(f"✅ Transcription done — {len(segments)} segments found.")

    # 3. Translate & write subtitles
    use_marianmt = (TRANSLATE_MODE == "marianmt")
    write_subtitles(segments, out_path, device, use_marianmt=use_marianmt)

    print(f"\n🎉 Done! Subtitles saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    device, compute_type = check_dependencies()

    # Collect video files (individual files or folders)
    video_files = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for f in sorted(os.listdir(arg)):
                if f.lower().endswith((".mp4", ".mkv")):
                    video_files.append(os.path.join(arg, f))
        elif os.path.isfile(arg):
            video_files.append(arg)
        else:
            print(f"⚠️  Not found: {arg}")

    if not video_files:
        print("❌ No .mp4 or .mkv files found.")
        sys.exit(1)

    print(f"\n📂 Found {len(video_files)} video(s) to process.")
    for i, vf in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]", end=" ")
        process_video(vf, device, compute_type)

    print("\n✅ All done!")
