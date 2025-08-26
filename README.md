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
# README language index

This repository contains two separate README files for clarity:

- `README.en.md` — English version
- `README.zh-CN.md` — 简体中文版本

Please open the file for the language you prefer. Keeping separate files avoids mixing languages and reduces reader confusion.

---

Note: if you want the repository to show a specific README by default on GitHub, rename that file to `README.md` or configure your documentation flow accordingly.
```bash

conda create -n aitype python=3.11 -y

conda activate aitype

```
