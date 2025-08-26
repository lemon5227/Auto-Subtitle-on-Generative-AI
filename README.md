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

# Auto-Subtitle-on-Generative-AI

Short description

This repository's default README is English. If you prefer reading in Chinese, click the "中文" link at the top to open the Chinese README (`README.zh-CN.md`).

This project is a demo for local subtitle generation and translation. It combines speech recognition (Whisper-style models) with translation models (from Hugging Face). The app supports extracting audio from local files or remote videos (e.g., YouTube), producing VTT subtitles, and optionally translating those subtitles into a target language. A lightweight web UI is included for uploads, downloads, model management and subtitle preview.

Key features:
- Fetch videos (background jobs) from URLs or use local files
- Transcribe audio into VTT using Whisper models
- Translate subtitles using translation models (optional bilingual output)
- Model management UI for downloading/removing models

See also: `README.en.md` (full English version) and `README.zh-CN.md` (中文完整版本).
```bash

conda create -n aitype python=3.11 -y

conda activate aitype

```
