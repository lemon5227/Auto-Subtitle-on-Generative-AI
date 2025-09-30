# macOS 安装指南

本应用已优化支持 macOS，包括 Apple Silicon (M1/M2/M3) Mac。

## 系统要求

- macOS 10.15+ (推荐 macOS 12+)
- Python 3.8+ (推荐 Python 3.11)
- 至少 8GB RAM (推荐 16GB+)

## 安装步骤

### 1. 安装 Homebrew (如果还没有)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装系统依赖
```bash
# 安装 ffmpeg (必需)
brew install ffmpeg

# 安装 Python (如果使用系统 Python)
brew install python@3.11
```

### 3. 安装 Python 环境
推荐使用 conda:
```bash
# 安装 Miniconda (如果还没有)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# 创建专用环境
conda create -n whisper-app python=3.11
conda activate whisper-app
```

### 4. 安装 PyTorch (Apple Silicon 优化版本)
```bash
# Apple Silicon Mac (M1/M2/M3)
pip install torch torchvision torchaudio

# Intel Mac
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. 安装应用依赖
```bash
cd Auto-Subtitle-on-Generative-AI
pip install -r requirements.txt
```

### 6. 运行应用
```bash
python app.py
```

## 性能优化

### Apple Silicon Mac (M1/M2/M3)
- 应用会自动检测并使用 MPS (Metal Performance Shaders) 进行 GPU 加速
- Large-v3 Turbo 模型在 M2/M3 Mac 上表现最佳
- 推荐使用 16GB+ 内存以获得最佳性能

### Intel Mac
- 应用将使用 CPU 进行推理
- 推荐使用 Small 或 Base 模型以获得合理的性能
- Distil-Whisper 模型可提供更快的转录速度

## 故障排除

### 常见问题

1. **ffmpeg 未找到**
   ```bash
   brew install ffmpeg
   ```

2. **PyTorch MPS 不可用**
   - 确保使用 macOS 12.3+ 和最新版 PyTorch
   - 某些旧版 macOS 不支持 MPS

3. **内存不足**
   - 使用较小的模型 (base, small)
   - 减少 batch_size 参数

4. **模型加载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间 (至少 5GB)

### 模型推荐

- **Apple Silicon Mac (16GB+ RAM)**: large-v3-turbo
- **Apple Silicon Mac (8GB RAM)**: small
- **Intel Mac**: base 或 distil-small.en

## 浏览器兼容性

在 macOS 上测试兼容：
- ✅ Safari 14+
- ✅ Chrome 88+
- ✅ Firefox 85+
- ✅ Edge 88+

## 性能基准

在 MacBook Pro M2 (16GB) 上：
- Large-v3 Turbo: ~2-3x 实时速度
- Small: ~5-6x 实时速度
- Base: ~4-5x 实时速度

在 MacBook Air M1 (8GB) 上：
- Small: ~3-4x 实时速度
- Base: ~2-3x 实时速度