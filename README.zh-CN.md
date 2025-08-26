# Auto-Subtitle-on-Generative-AI — 简体中文

简介

这是一个本地运行的字幕生成与翻译示例项目，结合了语音识别（Whisper 系列模型）和翻译模型（来自 Hugging Face）。该应用支持从本地文件或网络视频（例如 YouTube）提取音频、生成 VTT 字幕，并可将字幕翻译为目标语言。项目包含一个轻量级网页 UI，用于文件上传/下载、模型管理以及字幕预览与下载。

主要功能：
- 下载远程视频（后台任务）或使用本地文件
- 使用 Whisper 模型生成 VTT 字幕
- 使用翻译模型翻译字幕（可选双语显示）
- 模型管理界面（下载/删除）

依赖

系统依赖：
- ffmpeg（用于音频提取与转码）

Python 依赖（见 `requirements.txt`）：
- Flask
- openai-whisper 或 faster-whisper（可选）
- torch（根据是否使用 GPU 选择合适版本）
- transformers
- sentencepiece
- huggingface-hub
- yt-dlp

关于 faster-whisper 的说明：
- `faster-whisper` 是一个可选实现，在速度和内存占用方面比 `openai-whisper` 更优，尤其在 GPU 环境下。它使用优化过的解码器并能更高效利用 CUDA。
- 安装示例（可选）：

```bash
pip install faster-whisper
```

使用提示：使用 `faster-whisper` 时，需要在后端转录逻辑中引入并调用对应的 API。本仓库前端提供了 `use_faster` 开关，请确保后端处理该参数以启用 faster-whisper（如果已安装）。

快速开始

1）克隆仓库并进入目录

```bash
git clone https://github.com/<your-user>/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI
```

2）创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3）启动服务

```bash
python3 app.py
```

打开： http://127.0.0.1:5001/

使用说明

- 在网页 UI 中上传本地视频或粘贴 YouTube 链接并点击 Fetch（后台下载）。
- 视频准备好后在播放器预览，点击 Generate 开始转录。
- 转录完成后页面会显示 VTT 原始字幕，可编辑、翻译并下载。

后端接口速览

- POST /fetch -> 启动后台下载视频，返回 video_id
- GET /fetch/status?video_id=... -> 查询下载状态与服务器路径
- POST /upload -> 上传本地文件
- POST /extract_async -> 启动转录任务（参数示例 { video_path, model, use_faster, language }）
- GET /extract/status?job_id=... -> 查询转录任务状态与结果
- POST /translate -> 翻译 VTT 内容（参数示例 { vtt_content, source_lang, target_lang, video_path }）
- GET /models/status, POST /models/download, POST /models/delete -> 模型管理

常见问题

- 未找到 ffmpeg：请安装系统 ffmpeg。
- 模型下载失败或磁盘不足：检查磁盘空间并查看服务器日志（`server.log`）。
- 转录/翻译慢：使用更小的模型或在有 GPU 的机器上使用 faster-whisper / GPU 版本的 PyTorch。

许可与注意

本项目为示例，使用第三方模型或服务时请遵守相应条款。生产环境请考虑访问控制、模型下载许可与合规性问题。
