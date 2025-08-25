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

```bash
pip install -r requirements.txt
```

如果你需要 GPU 支持，请参考 PyTorch 官方安装命令并替换 `pip install torch`：

```bash
# CPU 示例（通用）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU 示例（根据 CUDA 版本选择，请参考 https://pytorch.org/）
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

4. 启动服务：

```bash
python3 app.py
```

然后打开浏览器访问：

```
http://127.0.0.1:5001/
```

## 前端使用流程

1. 在页面选择本地视频文件或在“URL”输入框粘贴 YouTube 等视频链接并点击 Fetch（后台下载）。
2. 下载完成后视频会自动加载到播放器。点击 Generate 开始提取字幕（选择模型）。
3. 页面会显示原始字幕（VTT），并可以选择翻译目标语言，翻译后可下载翻译后的 VTT。

## 重要 API（后端）

- `GET /models`：返回支持的 Whisper 模型列表
- `GET /models/status`：返回本地模型（Whisper / 翻译）当前状态（Ready / Not Downloaded / Downloading... / Error）
- `POST /models/download`：启动后台下载模型（传 `model_type` 和 `model_key`）
- `POST /models/delete`：删除本地模型缓存（Whisper 或翻译）
- `POST /fetch`：后台开始下载视频（返回 `video_id`）
- `GET /fetch/status?video_id=...`：查询视频下载状态与最终本地路径
- `GET /extract_subtitles?video_path=...&model=...`：对指定视频提取字幕（返回 VTT 内容）
- `POST /translate`：将 VTT 内容翻译为目标语言（需传 `vtt_content`, `source_lang`, `target_lang`, `video_path`）

## 常见问题与排查

- 后端报 `ffmpeg` 找不到：请先安装系统 `ffmpeg`（参见上文）。
- 模型列为 `Not Downloaded`：打开 Model Management，点击 Download，会在后台下载并在完成后显示 Ready。
- 前端在下载未完成就显示 Ready：后端已实现内存状态跟踪，若仍有误判请重启服务并观察后端日志。
- 翻译模型删除失败：Hugging Face 缓存目录可能位于 `$HF_HOME` 或 `~/.cache/huggingface/hub`，删除需要相应权限。

## 打包与发行建议

- 将 `requirements.txt` 与运行说明一并提供。为避免用户安装错误 `torch` 版本，建议在 README 中单独给出 CPU/GPU 的安装示例。
- 若要制作“一键运行”的镜像或 container，建议创建 `Dockerfile` 或 `environment.yml`（conda），并在镜像中预装 `ffmpeg` 与合适的 `torch` wheel。

## 许可与注意

本项目演示用途，模型与数据可能受第三方服务条款约束。请确保遵守 Hugging Face、YouTube 等服务的使用规则。



