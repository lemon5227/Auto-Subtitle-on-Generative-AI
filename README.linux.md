# Auto-Subtitle-on-Generative-AI

<div align="center">

## 🌐 Choose Your Language | 选择语言

| 🐧 Linux | 🇨🇳 中文 | 🇺🇸 English | 🍎 macOS | 🪟 WSL2 |
|:---:|:---:|:---:|:---:|:---:|
| **🐧 Linux 指南** | [📖 中文文档](README.zh-CN.md) | [📖 English](README.en.md) | [🍎 macOS 指南](README.macOS.md) | [🔧 WSL2 指南](README.wsl2.md) |
| **当前文档** | 完整中文说明 | Full English Guide | Apple Silicon 优化 | Windows 用户推荐 |

---

</div>

# 🐧 Linux 快速部署指南

本应用已全面优化支持 Linux 平台，包括 Ubuntu、Debian、CentOS、Fedora 等主流发行版。

## 🚀 一键丝滑部署

### 系统要求
- **Linux发行版**: Ubuntu 18.04+, Debian 10+, CentOS 7+, Fedora 30+, Arch Linux
- **Python**: 3.8+ (推荐 Python 3.11)
- **内存**: 至少 4GB RAM (推荐 8GB+)  
- **GPU**: 可选 NVIDIA GPU 用于 CUDA 加速
- **网络**: 稳定的网络连接用于下载模型

### 🎯 方法一：超级丝滑一键安装（最推荐）

只需一条命令，完全自动化部署：

```bash
curl -fsSL https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-linux.sh | bash
```

**或下载后执行：**
```bash
wget https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-linux.sh
chmod +x install-linux.sh
./install-linux.sh
```

**🎊 一键脚本功能:**
- 🔍 **智能检测**: 自动识别Linux发行版(Ubuntu/Debian/CentOS/Fedora/Arch)
- 📦 **系统依赖**: 自动安装ffmpeg、Python3、Git等必需组件
- 🐍 **Python环境**: 创建独立虚拟环境，避免依赖冲突
- 🚀 **GPU加速**: 自动检测NVIDIA GPU并安装CUDA版本PyTorch
- 📥 **依赖管理**: 安装所有Python依赖包
- ✅ **验证测试**: 完整性检查确保安装成功
- 🌐 **即时启动**: 安装完成后立即可用

### 🎯 方法二：Python智能启动器（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI

# 2. 运行智能启动器
python3 start.py
```

**智能启动器功能：**
- ✅ 检测系统环境和Python版本
- ✅ Linux平台自动安装系统依赖（sudo权限）
- ✅ 创建独立的虚拟环境
- ✅ 智能选择PyTorch版本（CUDA/CPU）
- ✅ 安装所有Python依赖包
- ✅ 一键启动Web服务

### 🛠️ 方法二：手动分步安装

#### 步骤1：安装系统依赖

**Ubuntu/Debian 系统：**
```bash
# 更新包管理器
sudo apt update

# 安装系统依赖
sudo apt install -y ffmpeg python3 python3-pip python3-venv git curl

# 安装conda (推荐)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

**CentOS/RHEL/Fedora 系统：**
```bash
# CentOS/RHEL
sudo yum install -y epel-release
sudo yum install -y ffmpeg python3 python3-pip git curl

# Fedora
sudo dnf install -y ffmpeg python3 python3-pip git curl

# 安装conda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### 步骤2：创建Python环境

```bash
# 创建专用环境
conda create -n whisper-app python=3.11 -y
conda activate whisper-app
```

#### 步骤3：安装PyTorch

**有NVIDIA GPU的用户（推荐）：**
```bash
# 检查CUDA版本
nvidia-smi

# 安装CUDA版PyTorch（根据CUDA版本选择）
# CUDA 11.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1+  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**仅CPU用户：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 步骤4：安装应用依赖

```bash
cd Auto-Subtitle-on-Generative-AI
pip install -r requirements.txt
```

#### 步骤5：启动服务

```bash
python app.py
```

## 🔧 故障排除

### 常见问题

**Q: ffmpeg 未找到**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS (需要EPEL源)
sudo yum install epel-release
sudo yum install ffmpeg

# 手动编译安装
wget https://ffmpeg.org/releases/ffmpeg-6.0.tar.xz
tar -xf ffmpeg-6.0.tar.xz
cd ffmpeg-6.0
./configure && make && sudo make install
```

**Q: CUDA支持检测**
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version

# 验证PyTorch CUDA支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Q: 权限问题**
```bash
# 为当前用户添加执行权限
chmod +x start.py
sudo chown -R $USER:$USER /path/to/Auto-Subtitle-on-Generative-AI
```

**Q: 端口被占用**
```bash
# 查看端口占用
sudo netstat -tlnp | grep :5001

# 结束占用进程
sudo kill -9 <PID>

# 或修改端口（在app.py中）
# app.run(host='0.0.0.0', port=5002, debug=True)
```

## 🚀 性能优化建议

### GPU加速配置
```bash
# 1. 确保NVIDIA驱动已安装
sudo ubuntu-drivers autoinstall

# 2. 安装CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 3. 验证安装
nvidia-smi
nvcc --version
```

### 内存优化
```bash
# 大模型运行建议
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 在Python中设置
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

### 启动优化
```bash
# 创建系统服务（可选）
sudo tee /etc/systemd/system/whisper-app.service > /dev/null <<EOF
[Unit]
Description=Whisper Subtitle App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/Auto-Subtitle-on-Generative-AI
ExecStart=/home/$USER/miniconda3/envs/whisper-app/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable whisper-app
sudo systemctl start whisper-app
```

## 🌟 访问应用

安装完成后，打开浏览器访问：
- **实时转录**: http://127.0.0.1:5001/realtime.html
- **文件处理**: http://127.0.0.1:5001/app.html

## 📞 技术支持

如遇问题，请：
1. 查看 [常见问题](#故障排除)
2. 提交 [GitHub Issue](https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI/issues)
3. 加入技术交流群

---
*Linux 平台已全面优化，享受丝滑的AI字幕生成体验！* 🚀