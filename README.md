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
- torch
- transformers
- sentencepiece
- huggingface-hub
- yt-dlp

> 注意：`torch` 建议按你的平台（CPU / CUDA 版本）选择合适的安装命令，下面有示例。

## 快速开始（建议使用 conda）

1. 创建并激活虚拟环境（示例使用 conda）：

```bash
conda create -n aitype python=3.11 -y
conda activate aitype
```

2. 安装系统依赖（Ubuntu/Debian 示例）：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

3. 安装 Python 依赖：

# Auto-Subtitle-on-Generative-AI / 自动字幕（中英双语）

## 简短说明（中文）

这是一个面向本地运行的字幕生成与翻译演示项目。它结合了语音识别（Whisper）与翻译模型（Hugging Face 提供的模型），支持从本地文件或网络视频（例如 YouTube）提取音轨、生成 VTT 格式字幕，并可将字幕翻译为目标语言。项目包含一个简洁的前端页面，用于上传/下载视频、管理模型与查看/下载字幕。

主要功能：
- 本地或远程视频下载（后台任务）
- 使用 Whisper 系列模型进行语音识别并生成 VTT 字幕
- 使用翻译模型将字幕翻译成指定语言（可选双语显示）
- 简单的模型管理（列出、下载、删除）

## Quick Overview (English)

This is a demo project for local subtitle generation and translation. It combines speech recognition (OpenAI/Whisper-style models) with translation models (from Hugging Face). The app supports extracting audio from local files or remote videos (e.g., YouTube), producing VTT subtitles, and optionally translating those subtitles into a target language. A lightweight web UI is included for uploads, downloads, model management and subtitle preview.

Key features:
- Fetch videos (background jobs) from URLs or use local files
- Transcribe audio into VTT using Whisper models
- Translate subtitles using translation models (optional bilingual output)
- Model management UI for downloading/removing models

## 必要依赖 / Requirements

系统依赖（system-level）：
- ffmpeg （用于音频提取/转码）

Python 依赖（参见 `requirements.txt`）：
- Flask
- openai-whisper 或 faster-whisper（可选）
- torch（根据是否使用 GPU 选择合适版本）
- transformers
- sentencepiece
- huggingface-hub
- yt-dlp

System note: install `ffmpeg` using your package manager (apt, brew, yum, etc.). For PyTorch, follow https://pytorch.org/ to pick the correct wheel for your CUDA version.

## 快速开始 / Quick start

1) 克隆并进入仓库 / Clone repository

```bash
git clone https://github.com/<your-user>/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI
```

2) 建议使用虚拟环境（conda 或 venv）/ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
# or with conda:
# conda create -n aitype python=3.11 -y && conda activate aitype
```

3) 安装依赖 / Install Python dependencies

```bash
pip install -r requirements.txt
```

4) 启动服务 / Run the web app

```bash
python3 app.py
```

访问： http://127.0.0.1:5001/

## 使用说明 / Usage

- 前端操作：上传本地视频或粘贴 YouTube 链接并点击 Fetch（后台下载）；下载完成后可在播放器预览并点击 Generate 开始转录。
- 生成完成后页面会显示 VTT 格式的原始字幕（可以编辑），可选择翻译目标语言并下载翻译结果。

## 后端 API 快速参考 / Backend API (short)

- POST /fetch -> 启动后台下载视频，返回 video_id
- GET /fetch/status?video_id=... -> 查询下载状态与本地路径
- POST /upload -> 上传本地文件（前端使用）
- POST /extract_async -> 启动后台转录任务，参数示例 { video_path, model, use_faster, language }
- GET /extract/status?job_id=... -> 查询转录任务状态与结果（可能包含 vtt 内容或路径）
- POST /translate -> 翻译 VTT 内容，参数示例 { vtt_content, source_lang, target_lang, video_path }
- GET /models/status, POST /models/download, POST /models/delete -> 模型管理

（具体细节请参阅源代码中 `app.py` 中的端点实现）

## 常见问题 / Troubleshooting

- ffmpeg 未安装：安装系统 ffmpeg。
- 模型下载失败或磁盘空间不足：检查磁盘空间，并查看服务器端日志（`server.log`）。
- 翻译/转录速度慢：启用更小的模型或在支持 GPU 的机器上使用 `faster-whisper`/GPU 版本的 PyTorch。

## 安全与许可 / Security & License

本仓库为演示用途。请遵守第三方模型与服务（Hugging Face、YouTube 等）的使用条款。对于生产部署，请考虑访问控制、模型下载许可以及合规性问题。

## 贡献 / Contributing

欢迎提交 issue 或 PR。如需帮助整合其他模型（如 WhisperX、faster-whisper、定制翻译模型），请在 issue 中描述你的需求与硬件环境。

---

如果你希望我把 README 的中文/英文描述进一步本地化（例如把 UI 文案改为中文，或增加更多运行示例），告诉我具体偏好，我会更新它。



