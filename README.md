# Auto-Subtitle-on-Generative-AI

<div align="center">

## 🌐 Choose Your Language | 选择语言

| 🇨🇳 中文 | 🇺🇸 English | 🍎 macOS | 🐧 Linux | 🪟 WSL2 | 🎮 AMD |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [📖 中文文档](README.zh-CN.md) | [📖 English](README.en.md) | [🍎 macOS 指南](README.macOS.md) | [🐧 Linux 指南](README.linux.md) | [🔧 WSL2 指南](README.wsl2.md) | [🎮 AMD 笔记本](README.amd.md) |
| 完整中文说明 | Full English Guide | Apple Silicon 优化 | 丝滑一键部署 | Windows 用户推荐 | AMD GPU 专用 |

---

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <strong>🎤 智能语音字幕生成器</strong><br>
  基于 Whisper Large-v3 Turbo 的实时转录和字幕生成系统
</p></p>

## ✨ 主要功能

- 🚀 **实时语音转录** - 支持麦克风实时监听和转录
- 🎯 **多模型支持** - Whisper Large-v3 Turbo, SenseVoice, Distil-Whisper
- 🤖 **Qwen3智能校对** - 集成最新Qwen3大模型，专业ASR纠错与翻译
- 🌍 **多语言识别** - 中文、英文、日语、韩语等
- 💻 **跨平台支持** - Windows, macOS (含 Apple Silicon), Linux
- 🎮 **智能GPU检测** - 自动检测并适配 NVIDIA CUDA / AMD ROCm / Apple MPS
- ⚡ **通用硬件加速** - 有GPU用GPU，无GPU智能回退CPU模式
- 📹 **视频处理** - 本地文件和 YouTube 视频下载转录
- 🔄 **字幕翻译** - 基于 Helsinki-NLP 和 Qwen3 的多语言翻译
- 💾 **多格式导出** - VTT, SRT, 纯文本格式
- 🎨 **现代化界面** - 基于 Tailwind CSS 的响应式设计

> 🆕 **最新更新**: Qwen3系列模型 - 更强的ASR纠错、专业字幕翻译、上下文感知理解！[查看详情](README.qwen3.md)

## 🖥️ 支持的平台和加速

| 平台 | CPU | NVIDIA GPU | AMD GPU | Apple GPU | 推荐配置 |
|------|-----|------------|---------|-----------|----------|
| **Windows** | ✅ | CUDA ✅ | ROCm ⚠️ | ❌ | RTX 3060+ |
| **macOS (Apple Silicon)** | ✅ | ❌ | ❌ | MPS ✅ | M1/M2/M3 16GB+ |
| **macOS (Intel)** | ✅ | ❌ | ❌ | ❌ | 8GB+ RAM |
| **Linux** | ✅ | CUDA ✅ | ROCm ✅ | ❌ | RTX 3060+ / RX 6600+ |
| **AMD 笔记本** | ✅ | ❌ | ROCm ✅ | ❌ | RX 6600 XT / Ryzen 7 |

> 🎮 **智能GPU检测**: 系统自动检测可用硬件，优先使用GPU加速，无GPU时智能回退CPU模式
> 🔴 **AMD GPU**: 支持 RX 6000/7000/5000 系列，ROCm 平台加速，笔记本用户推荐

## 🎮 智能GPU检测系统

本项目采用先进的智能GPU检测和适配系统，自动识别并优化各种硬件环境：

### 🔍 自动硬件检测
- **NVIDIA GPU**: 检测CUDA驱动，自动启用CUDA加速
- **AMD GPU**: 检测ROCm平台，支持RX 6000/7000/5000系列
- **Apple Silicon**: 检测MPS支持，Apple Silicon Mac优化
- **CPU模式**: 无GPU时自动回退，优化多线程性能

### ⚡ 智能设备选择
```bash
# 系统自动选择最佳设备
python start_smart.py  # 智能启动器会显示检测结果

# 示例输出:
# 🎯 选择设备: cuda
# 📊 设备信息: 🟢 NVIDIA GPU (RTX 3060, 6GB) - acceptable 性能
# 💡 使用 CUDA 加速获得最佳性能
```

### 🎯 性能优化建议
系统会根据检测到的硬件提供个性化建议：
- **NVIDIA用户**: 显存充足时推荐large模型
- **AMD用户**: 推荐small/base模型，ROCm优化
- **Apple Silicon**: medium模型性能最佳
- **CPU用户**: base模型获得最佳速度

### 🔧 环境验证工具
```bash
# 完整环境检测
python test_gpu_detection.py

# 快速GPU检测
python gpu_detector.py
```

本仓库提供完整的 demo 实现，支持生产环境部署（需要额外的安全加固和错误处理）。

## 🚀 快速开始

### 🎯 一键安装（推荐 - 适用于所有平台）
```bash
# 1. 克隆仓库
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI

# 2. 使用智能启动器（自动检查依赖、配置环境）
python start.py
```

**智能启动器功能**：
- ✅ 自动检测操作系统和Python环境
- ✅ 智能安装缺失的系统依赖（ffmpeg等）
- ✅ 创建独立的虚拟环境避免冲突
- ✅ 检测GPU硬件并配置最优加速
- ✅ 自动下载必需模型文件
- ✅ 一键启动Web服务

### 🚀 平台特定丝滑部署

#### 🐧 Linux 超级丝滑一键安装
```bash
# 只需一条命令，全自动部署！
curl -fsSL https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-linux.sh | bash
```

#### 🪟 Windows WSL2 专用部署
```bash
# WSL2 环境优化部署
curl -fsSL https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-wsl2.sh | bash
```

#### 🎮 AMD 笔记本专用部署
```bash
# AMD GPU 笔记本用户专用一键部署
curl -fsSL https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-amd.sh | bash
```

| 平台 | 部署方案 | 特色优化 | 安装时间 |
|------|----------|----------|----------|
| 🐧 **Linux** | [🚀 一键脚本](README.linux.md) | 检测发行版 • 自动装依赖 • GPU加速 | ~5分钟 |
| 🎮 **AMD 笔记本** | [🔴 AMD专用](README.amd.md) | ROCm支持 • AMD GPU优化 • 笔记本适配 | ~6分钟 |
| 🪟 **WSL2** | [🔧 WSL2专用](README.wsl2.md) | GPU支持 • 环境优化 • 兼容性修复 | ~6分钟 |
| 🍎 **macOS** | [🍎 Apple优化](README.macOS.md) | Apple Silicon • MPS加速 • Homebrew | ~8分钟 |
| 🪟 **Windows** | [参考通用步骤](#🚀-快速开始) | CUDA支持 • 虚拟环境 | ~10分钟 |

> 💡 **最佳体验**: 
> - **Linux** 原生用户: 使用一键脚本，性能最佳
> - **AMD 笔记本** 用户: 使用AMD专用脚本，ROCm GPU加速
> - **Windows** 用户: 推荐 WSL2 方案，体验接近原生 Linux
> - **macOS** 用户: 使用专用指南，Apple Silicon 优化

## 📱 功能演示

### 实时转录界面
- **访问地址**: http://127.0.0.1:5001/realtime.html
- **功能**: 实时语音识别、多语言支持、字幕导出

### 文件处理界面  
- **访问地址**: http://127.0.0.1:5001/app.html
- **功能**: 视频上传、批量转录、翻译、模型管理

## 📋 系统依赖

### 必需组件
- **ffmpeg**: 音频视频处理核心
- **Python 3.8+**: 推荐 3.11 版本
- **Git**: 代码克隆和版本管理

### 快速安装系统依赖

**Ubuntu/Debian Linux:**
```bash
sudo apt update && sudo apt install -y ffmpeg python3 python3-pip git
```

**CentOS/RHEL/Fedora Linux:**
```bash
# CentOS/RHEL
sudo yum install -y epel-release && sudo yum install -y ffmpeg python3 python3-pip git

# Fedora  
sudo dnf install -y ffmpeg python3 python3-pip git
```

**macOS:**
```bash
brew install ffmpeg python@3.11 git
```

**Windows:**
```bash
# 使用 Chocolatey
choco install ffmpeg python git

# 或下载安装包
# Python: https://python.org/downloads
# FFmpeg: https://ffmpeg.org/download.html
# Git: https://git-scm.com/downloads
```

### Python 依赖（自动安装）
主要组件（详见 `requirements.txt`）：
- **Flask**: Web 服务框架
- **openai-whisper**: 语音识别核心
- **torch**: PyTorch 深度学习框架  
- **transformers**: 翻译模型支持

## 🍎 macOS 用户快速配置指南

### 系统要求
- macOS 10.15+ (推荐 macOS 12+)
- Python 3.8+ (推荐 Python 3.11)
- 至少 8GB RAM (推荐 16GB+)

### 1. 安装 Homebrew（如果没有）
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装系统依赖
```bash
# 安装 ffmpeg（必需）
brew install ffmpeg

# 安装 Python（可选，如果使用系统 Python）
brew install python@3.11
```

### 3. 配置 Python 环境
**推荐使用 conda：**
```bash
# 下载并安装 Miniconda（如果没有）
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# 创建专用环境
conda create -n whisper-app python=3.11 -y
conda activate whisper-app
```

### 4. 安装 PyTorch（重要：选择正确版本）
```bash
# Apple Silicon Mac (M1/M2/M3) - 支持 MPS 加速
pip install torch torchvision torchaudio

# Intel Mac - CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. 克隆项目并安装依赖
```bash
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI

# 安装应用依赖
pip install -r requirements.txt
```

### 6. 启动应用
```bash
# 使用跨平台启动器（推荐）
python start.py

# 或直接启动
python app.py
```

### 7. 访问应用
- 主界面：http://127.0.0.1:5001/app.html
- 实时转录：http://127.0.0.1:5001/realtime.html

### macOS 性能优化建议

**Apple Silicon Mac (M1/M2/M3)：**
- ✅ 自动使用 MPS (Metal Performance Shaders) GPU 加速
- 推荐模型：`large-v3-turbo` (16GB+ RAM) 或 `small` (8GB RAM)
- 预期性能：2-3x 实时转录速度

**Intel Mac：**
- 使用 CPU 模式，性能较慢但稳定
- 推荐模型：`base` 或 `distil-small.en`
- 建议使用较小的音频分块以减少内存使用

### 故障排除
- **ffmpeg 未找到**：`brew install ffmpeg`
- **PyTorch MPS 不可用**：确保使用 macOS 12.3+ 和最新版 PyTorch
- **内存不足**：使用更小的模型或减少批处理大小
- **模型下载失败**：检查网络连接和磁盘空间（至少 5GB）

详细的 macOS 配置说明请参考：[README.macOS.md](README.macOS.md)
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

## 🍎 macOS Quick Setup Guide

### System Requirements
- macOS 10.15+ (recommended macOS 12+)
- Python 3.8+ (recommended Python 3.11)
- At least 8GB RAM (recommended 16GB+)

### 1. Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install System Dependencies
```bash
# Install ffmpeg (required)
brew install ffmpeg

# Install Python (optional, if using system Python)
brew install python@3.11
```

### 3. Setup Python Environment
**Recommended using conda:**
```bash
# Download and install Miniconda (if not installed)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Create dedicated environment
conda create -n whisper-app python=3.11 -y
conda activate whisper-app
```

### 4. Install PyTorch (Important: Choose correct version)
```bash
# Apple Silicon Mac (M1/M2/M3) - MPS acceleration support
pip install torch torchvision torchaudio

# Intel Mac - CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Clone and Install Dependencies
```bash
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI

# Install application dependencies
pip install -r requirements.txt
```

### 6. Launch Application
```bash
# Use cross-platform launcher (recommended)
python start.py

# Or launch directly
python app.py
```

### 7. Access Application
- Main Interface: http://127.0.0.1:5001/app.html
- Real-time Transcription: http://127.0.0.1:5001/realtime.html

### macOS Performance Optimization

**Apple Silicon Mac (M1/M2/M3):**
- ✅ Automatic MPS (Metal Performance Shaders) GPU acceleration
- Recommended models: `large-v3-turbo` (16GB+ RAM) or `small` (8GB RAM)
- Expected performance: 2-3x real-time transcription speed

**Intel Mac:**
- CPU mode, slower but stable performance
- Recommended models: `base` or `distil-small.en`
- Suggest using smaller audio chunks to reduce memory usage

For detailed macOS configuration instructions, see: [README.macOS.md](README.macOS.md)

## Quick start

1) Clone repository

```bash
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
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

## 🔧 故障排除

### 常见问题
- **ffmpeg 未找到**: 安装系统 ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)
- **模型下载失败**: 检查磁盘空间（至少 5GB）和网络连接
- **转录速度慢**: 使用更小的模型或 GPU 加速
- **内存不足**: 减少批处理大小或使用 CPU 优化模式

### 性能基准
| 设备类型 | 模型 | 预期速度 | 推荐配置 |
|----------|------|----------|----------|
| RTX 4090 | large-v3-turbo | ~5-8x 实时 | 24GB VRAM |
| RTX 3060 | large-v3-turbo | ~3-5x 实时 | 12GB VRAM |
| M2 Max | large-v3-turbo | ~2-3x 实时 | 32GB RAM |
| M1 | small | ~3-4x 实时 | 16GB RAM |
| CPU (Intel) | base | ~1-2x 实时 | 16GB RAM |

## 🤝 贡献指南

我们欢迎 PR 和 Issue！如果需要集成其他模型（WhisperX、自定义翻译模型），请开 Issue 详细描述您的环境和需求。

### 开发环境
```bash
# 克隆仓库
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI

# 安装开发依赖
pip install -r requirements.txt
pip install black flake8 pytest  # 代码格式和测试

# 运行测试
python test_turbo.py
```

## 📄 许可和注意事项

本项目采用 MIT 许可证。请确保遵守第三方模型/服务的条款（Hugging Face、YouTube）。用于生产环境时，请考虑访问控制、许可合规和安全加固。

## ⭐ 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 模型框架
- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - SenseVoice 支持
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 性能优化

---

<p align="center">
  如果这个项目对您有帮助，请给个 ⭐ Star！
</p>
