# Auto-Subtitle-on-Generative-AI

<div align="center">

## 🌐 Choose Your Language | 选择语言

| 🇺🇸 English | 🇨🇳 中文 | 🍎 macOS | 🐧 Linux | 🪟 WSL2 |
|:---:|:---:|:---:|:---:|:---:|
| **📖 English Guide** | [📖 中文文档](README.zh-CN.md) | [🍎 macOS Guide](README.macOS.md) | [🐧 Linux Guide](README.linux.md) | [🔧 WSL2 Guide](README.wsl2.md) |
| **Current Document** | Complete Chinese Guide | Apple Silicon Optimized | One-Click Deployment | Windows Users Recommended |

---

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <strong>🎤 AI Intelligent Subtitle Generator</strong><br>
  Real-time transcription and subtitle generation system based on Whisper Large-v3 Turbo
</p>

## 📖 Project Overview

This is a demo project for local subtitle generation and translation. It combines speech recognition (Whisper-style models) with translation models (from Hugging Face). The app supports extracting audio from local files or remote videos (e.g., YouTube), producing VTT subtitles, and optionally translating those subtitles into a target language. A lightweight web UI is included for uploads, downloads, model management and subtitle preview.

Key features:
- Fetch videos (background jobs) from URLs or use local files
- Transcribe audio into VTT using Whisper models
- Translate subtitles using translation models (optional bilingual output)
- Model management UI for downloading/removing models

Requirements

System-level:
- ffmpeg (for audio extraction and transcode)

Python dependencies (see `requirements.txt`):
- Flask
- openai-whisper or faster-whisper (optional)
- torch (choose the correct wheel for your platform/CUDA)
- transformers
- sentencepiece
- huggingface-hub
- yt-dlp

Note about faster-whisper:
- `faster-whisper` is an alternative implementation that can be significantly faster and more memory-efficient than `openai-whisper`, especially when using GPU. It uses optimized decoders and can leverage CUDA more effectively.
- To install faster-whisper (optional):

```bash
pip install faster-whisper
```

Usage note: when using `faster-whisper`, adapt the server-side transcription code to import and call its API. The repository supports selecting `use_faster` in the frontend which you should wire to your server-side handler to enable faster-whisper if available.

Quick start

1) Clone repository

```bash
git clone https://github.com/<your-user>/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI
```

2) Recommended: use conda (preferred)

If you use conda, the steps below create an environment, install ffmpeg from conda-forge and install PyTorch via conda (recommended for GPU users):

```bash
# create and activate conda env
conda create -n aitype python=3.11 -y
conda activate aitype

# install ffmpeg
conda install -c conda-forge ffmpeg -y

# install pytorch (CPU example)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# then install other python deps
pip install -r requirements.txt
```

For GPU users, select the correct cudatoolkit and package set from https://pytorch.org/. Example (CUDA 11.8):

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
```

If you prefer venv, you can still use:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3) Start the app

```bash
python3 app.py
```

Visit: http://127.0.0.1:5001/

Usage

- Use the web UI to upload a local video file or paste a YouTube link and click Fetch to download in the background.
- After the video is ready, preview it in the player and click Generate to start transcription.
- The app will return VTT subtitles which you can edit, translate, and download.

Backend API (short)

- POST /fetch -> start background download, returns video_id
- GET /fetch/status?video_id=... -> check download status and server path
- POST /upload -> upload a local file
- POST /extract_async -> start transcription job (payload: { video_path, model, use_faster, language })
- GET /extract/status?job_id=... -> check transcription job status / result
- POST /translate -> translate VTT content (payload: { vtt_content, source_lang, target_lang, video_path })
- GET /models/status, POST /models/download, POST /models/delete -> model management

Troubleshooting

- ffmpeg not found: install system ffmpeg.
- Model download fails: check disk space and server logs (`server.log`).
- Slow transcription/translation: use smaller models or a machine with GPU and faster-whisper/PyTorch GPU build.

License & Notes

This repo is for demo purposes only. Observe third-party model/service terms (Hugging Face, YouTube). For production use, consider access control and licensing.
