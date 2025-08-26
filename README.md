# Auto-Subtitle-on-Generative-AI

这是一个基于 Whisper / Hugging Face 的本地字幕生成与翻译示例工程。主要功能：

- 从本地或视频 URL（如 YouTube）下载视频
- 用 Whisper（多种模型）做语音识别并生成 VTT 字幕
- 使用 Helsinki-NLP 翻译模型将字幕翻译成目标语言
- 模型管理（查看、本地下载、删除）

本仓库旨在做为可跑通的 demo；如果要在生产或更大规模使用，需要对模型下载、权限和错误处理做更多硬化。

## 必要依赖

系统依赖：
- `ffmpeg`（音频提取与转码）

Python 依赖（见 `requirements.txt`）：
- Flask
- openai-whisper
<!-- Language selector: default is English. Click a link to switch. -->
<p align="right">Language: <strong>English</strong> | <a href="README.zh-CN.md">中文</a></p>

# Auto-Subtitle-on-Generative-AI — English (Full)

Short description

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

If you use conda, the following steps will create an environment, install system-level packages (ffmpeg) from conda-forge, and install Python dependencies. Conda is recommended because it simplifies installing PyTorch and matching CUDA toolkits.

```bash
# create and activate conda env
conda create -n aitype python=3.11 -y
conda activate aitype

# install ffmpeg from conda-forge
conda install -c conda-forge ffmpeg -y

# install pytorch (CPU example) via conda (replace with appropriate cudatoolkit for GPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# install other python deps
pip install -r requirements.txt
```

Notes on PyTorch/GPU: For GPU support, pick the correct conda command from https://pytorch.org/ (select your OS, package manager=conda, and CUDA version). Example (GPU, CUDA 11.8):

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
```

If you prefer a plain venv instead of conda, you can still use:

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

Front-end quick workflow

1. Upload a local video file or paste a YouTube link and click Fetch (background download).
2. After download completes, the video will be available in the player. Click Generate to start transcription (choose model/options).
3. The UI will show VTT subtitles; you can edit, translate, and download them.

Backend API (reference)

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

Notes & License

This repository is a demo. Please ensure you follow third-party model/service terms (Hugging Face, YouTube). For production use, consider access control, licensing, and compliance.

Contributing

PRs and issues welcome. If you want help integrating other models (WhisperX, custom translation models), open an issue with details about your environment and goals.

```bash

conda create -n aitype python=3.11 -y

conda activate aitype

```
