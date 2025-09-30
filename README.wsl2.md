# WSL2 环境部署指南

WSL2 (Windows Subsystem for Linux 2) 环境下的 AI 字幕生成器部署完全指南。

## 🖥️ WSL2 环境说明

WSL2 提供了在 Windows 上运行 Linux 环境的能力，但 GPU 支持需要特殊配置。

### 系统要求
- **Windows 11** 或 **Windows 10** (版本 21H2 或更高)
- **WSL2** 已安装并启用
- **Ubuntu 20.04+** 或 **Debian 11+** (推荐 Ubuntu 22.04)
- 内存：至少 8GB RAM (推荐 16GB+)
- **可选**: NVIDIA GPU (RTX 20/30/40 系列)

## 🚀 方法一：WSL2 一键部署脚本

### 超级简单一键安装

```bash
# 在 WSL2 Ubuntu 环境中运行
curl -fsSL https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-wsl2.sh | bash
```

## 🛠️ 方法二：手动详细部署

### 步骤 1: 准备 WSL2 环境

#### 1.1 安装 WSL2 (在 Windows PowerShell 管理员模式)
```powershell
# 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 重启 Windows 后继续
wsl --set-default-version 2

# 安装 Ubuntu 22.04
wsl --install -d Ubuntu-22.04
```

#### 1.2 配置 WSL2 资源限制
在 Windows 用户目录下创建 `.wslconfig` 文件：

```ini
# C:\Users\<YourUsername>\.wslconfig
[wsl2]
memory=8GB          # 分配给 WSL2 的内存
processors=4        # CPU 核心数
swap=2GB           # 交换文件大小
localhostForwarding=true
```

重启 WSL2：
```bash
# 在 Windows CMD/PowerShell 中
wsl --shutdown
wsl
```

### 步骤 2: WSL2 系统依赖安装

#### 2.1 更新系统
```bash
# 在 WSL2 Ubuntu 中运行
sudo apt update && sudo apt upgrade -y
```

#### 2.2 安装基础工具
```bash
# 安装系统依赖
sudo apt install -y \
    python3 python3-pip python3-venv \
    git curl wget \
    ffmpeg \
    build-essential \
    software-properties-common
```

#### 2.3 安装 Python 环境管理
```bash
# 安装 conda (推荐)
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
rm miniconda.sh
```

### 步骤 3: GPU 支持配置 (可选但推荐)

#### 3.1 检查 Windows 端 NVIDIA 驱动
```powershell
# 在 Windows PowerShell 中检查
nvidia-smi
```

确保 NVIDIA 驱动版本 >= 470.76

#### 3.2 WSL2 中不需要安装 CUDA Toolkit
⚠️ **重要**: WSL2 环境下**不要**在 Linux 中安装 CUDA Toolkit，使用 Windows 端的即可。

#### 3.3 验证 GPU 支持
```bash
# 在 WSL2 中检查
ls -la /usr/lib/wsl/lib/
# 应该看到 libcuda.so, libcudart.so 等文件
```

### 步骤 4: 部署应用

#### 4.1 克隆项目
```bash
cd ~
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI
```

#### 4.2 创建 Python 环境
```bash
# 使用 conda 创建环境
conda create -n whisper-app python=3.11 -y
conda activate whisper-app
```

#### 4.3 安装 PyTorch (WSL2 专用)
```bash
# WSL2 环境下的 PyTorch 安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4.4 安装应用依赖
```bash
pip install -r requirements.txt
```

#### 4.5 验证 GPU 支持 (可选)
```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print('GPU test:', torch.cuda.get_device_properties(0))
"
```

### 步骤 5: 启动服务

#### 5.1 启动应用
```bash
# 激活环境
conda activate whisper-app

# 启动服务
python app.py
```

#### 5.2 访问应用
在 Windows 浏览器中访问：
- **实时转录**: http://localhost:5001/realtime.html  
- **文件处理**: http://localhost:5001/app.html

## 🔧 WSL2 特殊配置和优化

### GPU 内存优化
```bash
# 设置环境变量避免 CUDA 内存问题
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128' >> ~/.bashrc
echo 'export CUDA_LAUNCH_BLOCKING=1' >> ~/.bashrc
source ~/.bashrc
```

### 创建 WSL2 专用启动脚本
```bash
cat > start_wsl2.py << 'EOF'
#!/usr/bin/env python3
"""
WSL2 专用启动脚本 - 解决 WSL2 环境特殊问题
"""
import os
import sys
import subprocess

# WSL2 环境优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("🔧 WSL2 优化启动器")
print("🚀 针对 WSL2 环境优化的 AI 字幕生成器")
print("-" * 50)

# 检查 WSL2 环境
if 'microsoft' not in open('/proc/version').read().lower():
    print("⚠️  警告: 似乎不在 WSL2 环境中")

try:
    subprocess.run([sys.executable, "app.py"], check=True)
except KeyboardInterrupt:
    print("\n👋 服务已停止")
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    print("\n💡 如果是 CUDA 错误，尝试:")
    print("   export CUDA_VISIBLE_DEVICES=''")
    print("   python app.py")
EOF

chmod +x start_wsl2.py
```

### 文件系统优化
```bash
# WSL2 中推荐在 Linux 文件系统中操作，性能更好
cd /home/$USER/Auto-Subtitle-on-Generative-AI

# 避免在 Windows 文件系统 (/mnt/c/) 中运行，性能较差
```

## 🚨 常见问题和解决方案

### Q1: CUDA 错误 "no kernel image is available"
```bash
# 解决方案1: 强制使用 CPU
export CUDA_VISIBLE_DEVICES=''
python app.py

# 解决方案2: 重新安装兼容的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Q2: 内存不足错误
```bash
# 增加 WSL2 内存限制，编辑 Windows 端 .wslconfig
# [wsl2]
# memory=12GB

# 重启 WSL2
wsl --shutdown
```

### Q3: 端口访问问题
```bash
# Windows 防火墙可能阻止访问，添加防火墙规则
# 或者使用 localhost 而不是 127.0.0.1
```

### Q4: 性能优化
```bash
# 1. 确保项目在 Linux 文件系统中 (/home/user/)
# 2. 使用 conda 环境管理
# 3. 启用 Windows Terminal 性能模式
# 4. 关闭不必要的 Windows 服务
```

### Q5: 模型下载慢
```bash
# 配置 HuggingFace 镜像 (中国用户)
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
```

## 📝 WSL2 部署清单

- [ ] Windows 11/10 已更新到最新版本
- [ ] WSL2 已安装并配置
- [ ] Ubuntu 22.04 在 WSL2 中运行
- [ ] `.wslconfig` 已配置足够内存
- [ ] 系统依赖已安装 (ffmpeg, python3, git)
- [ ] Conda 环境已创建
- [ ] PyTorch 已安装 (CUDA 或 CPU 版本)
- [ ] 应用依赖已安装
- [ ] 端口 5001 可访问
- [ ] GPU 支持已验证 (可选)

## 🎯 性能建议

### 最佳配置
- **内存**: 12GB+ 分配给 WSL2
- **存储**: 项目放在 Linux 文件系统 (`/home/user/`)
- **网络**: 使用 `localhost` 访问服务
- **GPU**: 如遇问题优先使用 CPU 模式

### 推荐工作流
1. 开发调试: CPU 模式，稳定快速
2. 生产使用: GPU 模式，性能最佳  
3. 批量处理: 大内存配置 + GPU 加速

---

🎉 **WSL2 部署完成！享受在 Windows 上的 Linux AI 体验！**