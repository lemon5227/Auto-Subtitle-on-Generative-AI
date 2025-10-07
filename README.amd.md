# AMD 笔记本部署指南

专为 AMD GPU 笔记本用户设计的 AI 字幕生成器完整部署方案。

## 🎮 AMD GPU 支持说明

### AMD GPU 架构支持
- ✅ **RDNA2/RDNA3**: RX 6000/7000 系列 (推荐)
- ✅ **RDNA**: RX 5000 系列 (良好支持)  
- ⚠️ **GCN**: RX 400/500 系列 (有限支持)
- ✅ **APU**: Ryzen 集显 (基础支持)

### 系统要求
- **操作系统**: Linux (Ubuntu 20.04+, 推荐22.04) 
- **内存**: 至少 8GB RAM (推荐 16GB+)
- **存储**: 至少 20GB 可用空间
- **网络**: 稳定连接用于模型下载

## 🚀 AMD 专用一键部署

### 方法一：AMD 优化安装脚本

```bash
# AMD 笔记本专用一键安装
curl -fsSL https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-amd.sh | bash
```

### 方法二：手动详细部署

#### 步骤 1: 系统准备

**更新系统包:**
```bash
sudo apt update && sudo apt upgrade -y
```

**安装基础依赖:**
```bash
sudo apt install -y \
    python3 python3-pip python3-venv \
    git curl wget \
    ffmpeg \
    build-essential \
    software-properties-common \
    dkms
```

#### 步骤 2: AMD GPU 驱动安装

**检查 AMD GPU 信息:**
```bash
lspci | grep -i amd
lsmod | grep amdgpu
```

**安装 AMD GPU 驱动:**
```bash
# Ubuntu 官方驱动 (推荐)
sudo apt install -y mesa-vulkan-drivers xserver-xorg-video-amdgpu

# 或安装 AMD 官方驱动
wget https://repo.radeon.com/amdgpu-install/22.40.5/ubuntu/jammy/amdgpu-install_5.4.50405-1_all.deb
sudo dpkg -i amdgpu-install_5.4.50405-1_all.deb
sudo apt update
sudo apt install -y amdgpu-dkms
```

#### 步骤 3: ROCm 平台安装 (GPU 加速)

**添加 ROCm 仓库:**
```bash
# 添加 ROCm APT 仓库
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
```

**安装 ROCm:**
```bash
# 核心 ROCm 组件
sudo apt install -y rocm-dev rocm-libs hip-dev

# 添加用户到 render 组
sudo usermod -a -G render,video $USER

# 重新登录或重启以应用组权限
```

**验证 ROCm 安装:**
```bash
# 检查 ROCm 信息
/opt/rocm/bin/rocminfo

# 检查设备
ls /dev/kfd /dev/dri/render*
```

#### 步骤 4: Python 环境配置

**创建虚拟环境:**
```bash
cd ~
git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
cd Auto-Subtitle-on-Generative-AI

# 使用 venv
python3 -m venv venv_amd
source venv_amd/bin/activate

# 或使用 conda
# conda create -n whisper-amd python=3.11 -y
# conda activate whisper-amd
```

#### 步骤 5: PyTorch ROCm 版本安装

**安装 PyTorch ROCm 版本:**
```bash
# ROCm 5.7 对应的 PyTorch 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 验证 ROCm 支持
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')  # ROCm 使用 cuda API
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

#### 步骤 6: 应用依赖安装

```bash
# 安装应用依赖
pip install -r requirements.txt

# AMD 优化的额外依赖
pip install accelerate optimum[onnxruntime]
```

## 🔧 AMD 特殊配置

### 环境变量优化

创建 AMD 专用启动脚本:
```bash
cat > start_amd.py << 'EOF'
#!/usr/bin/env python3
"""
AMD GPU 专用启动脚本 - AI 字幕生成器
🎮 针对 AMD GPU 和 ROCm 平台优化
"""
import os
import sys
import subprocess

def setup_amd_environment():
    """配置 AMD GPU 优化环境"""
    print("🎮 AMD GPU 优化启动器")
    print("🚀 ROCm 平台 AI 字幕生成器")
    print("=" * 50)
    
    # AMD GPU 优化环境变量
    amd_env = {
        # ROCm 设置
        'HSA_OVERRIDE_GFX_VERSION': '10.3.0',  # 兼容性设置
        'ROCM_PATH': '/opt/rocm',
        'HIP_VISIBLE_DEVICES': '0',  # 使用第一个 GPU
        
        # PyTorch 优化
        'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:128',
        'TOKENIZERS_PARALLELISM': 'false',
        
        # 内存优化
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4',
        
        # AMD GPU 特定优化
        'AMD_LOG_LEVEL': '1',
        'HIP_LAUNCH_BLOCKING': '0',
    }
    
    for key, value in amd_env.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    print("\n🔍 检查 AMD GPU 支持...")
    check_amd_gpu()

def check_amd_gpu():
    """检查 AMD GPU 环境"""
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        
        # 检查 ROCm/HIP 支持
        gpu_available = torch.cuda.is_available()
        print(f"   GPU 可用: {'✅ 是' if gpu_available else '❌ 否'}")
        
        if gpu_available:
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
            
            # 简单 GPU 测试
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.matmul(x, x)
                print("   GPU 测试: ✅ 通过")
                del x, y
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   GPU 测试: ❌ 失败 - {e}")
        else:
            print("   💡 将使用 CPU 模式")
            
    except ImportError:
        print("   ❌ PyTorch 未安装")

def start_application():
    """启动应用"""
    print("\n🌐 启动 AI 字幕生成器...")
    print("   📺 实时转录: http://localhost:5001/realtime.html")
    print("   🎬 文件处理: http://localhost:5001/app.html")
    print("\n🔔 智能GPU适配提示:")
    print("   - 系统会自动检测AMD/NVIDIA/Apple GPU")
    print("   - AMD GPU需要ROCm支持，无ROCm会回退CPU模式")
    print("   - 大模型可能需要 8GB+ 显存")
    print("   - 按 Ctrl+C 停止服务")
    print("\n" + "-" * 50)
    
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n💡 通用GPU故障排除:")
        print("   1. 检查依赖: pip install -r requirements.txt")
        print("   2. AMD用户: 确保ROCm已安装")
        print("   3. NVIDIA用户: 检查CUDA驱动")
        print("   4. 强制CPU模式: export CUDA_VISIBLE_DEVICES=-1")
        print("   5. 使用智能启动器: python start_smart.py")

if __name__ == "__main__":
    setup_amd_environment()
    start_application()
EOF

chmod +x start_amd.py
```

### GPU 内存管理

**创建 AMD 专用配置:**
```bash
# AMD GPU 内存配置
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## 🔍 验证和测试

### 完整环境检查
```bash
# 创建检查脚本
cat > check_amd_env.py << 'EOF'
#!/usr/bin/env python3
"""AMD GPU 环境完整检查脚本"""

import os
import subprocess
import sys

def check_system():
    print("🖥️ 系统信息检查")
    print("-" * 30)
    
    # 检查 AMD GPU
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        amd_gpus = [line for line in result.stdout.split('\n') if 'AMD' in line and ('VGA' in line or 'Display' in line)]
        if amd_gpus:
            print("✅ AMD GPU 检测:")
            for gpu in amd_gpus:
                print(f"   {gpu.strip()}")
        else:
            print("❌ 未检测到 AMD GPU")
    except:
        print("❌ GPU 检查失败")
    
    # 检查 ROCm
    rocm_path = '/opt/rocm'
    if os.path.exists(rocm_path):
        print(f"✅ ROCm 安装路径: {rocm_path}")
        
        # 检查 rocminfo
        try:
            result = subprocess.run([f'{rocm_path}/bin/rocminfo'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ rocminfo 可用")
            else:
                print("⚠️ rocminfo 执行异常")
        except:
            print("❌ rocminfo 不可用")
    else:
        print("❌ ROCm 未安装")

def check_pytorch():
    print("\n🔥 PyTorch ROCm 检查")
    print("-" * 30)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA/ROCm: {'可用' if torch.cuda.is_available() else '不可用'}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 简单测试
        if torch.cuda.is_available():
            try:
                x = torch.randn(1000, 1000)
                x_gpu = x.cuda()
                result = torch.matmul(x_gpu, x_gpu)
                print("✅ GPU 计算测试: 通过")
                del x, x_gpu, result
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU 计算测试: 失败 - {e}")
    except ImportError:
        print("❌ PyTorch 未安装")
    except Exception as e:
        print(f"❌ PyTorch 检查失败: {e}")

def check_dependencies():
    print("\n📦 依赖检查")
    print("-" * 30)
    
    deps = ['flask', 'whisper', 'torch', 'transformers']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} 未安装")

if __name__ == "__main__":
    print("🎮 AMD GPU AI 字幕生成器 - 环境检查")
    print("=" * 60)
    check_system()
    check_pytorch()
    check_dependencies()
    print("\n" + "=" * 60)
    print("💡 如有问题，请参考 AMD 部署指南进行故障排除")
EOF

python check_amd_env.py
```

## 🚨 常见问题和解决方案

### Q1: ROCm 检测不到 GPU
```bash
# 检查内核模块
lsmod | grep amdgpu

# 重新加载驱动
sudo modprobe -r amdgpu
sudo modprobe amdgpu

# 检查设备权限
ls -la /dev/kfd /dev/dri/render*
```

### Q2: PyTorch 无法使用 GPU
```bash
# 检查 ROCm 环境变量
echo $ROCM_PATH
echo $HSA_OVERRIDE_GFX_VERSION

# 重新安装 ROCm 版 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Q3: 内存不足错误
```bash
# 减少内存使用
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:64
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# 使用较小模型
# 在应用中选择 base 或 small 模型
```

### Q4: 性能较慢
```bash
# 确保使用高性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 检查 GPU 频率
sudo cat /sys/class/drm/card0/device/pp_dpm_sclk
```

## 🎯 AMD 性能优化建议

### 硬件配置推荐
- **最佳**: RX 6700 XT / RX 7800 XT (12GB+ 显存)
- **推荐**: RX 6600 XT / RX 7600 (8GB 显存)  
- **可用**: RX 5500 XT / APU (4GB+ 显存)

### 软件配置优化
```bash
# 创建性能配置脚本
cat > optimize_amd.sh << 'EOF'
#!/bin/bash
# AMD GPU 性能优化脚本

echo "🚀 应用 AMD GPU 性能优化..."

# CPU 调度器优化
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# GPU 电源管理
echo high | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level

# 内存优化
echo 1 | sudo tee /proc/sys/vm/drop_caches

echo "✅ AMD 优化完成"
EOF

chmod +x optimize_amd.sh
```

## 🌟 启动应用

### 使用 AMD 优化启动器
```bash
# 激活环境
source venv_amd/bin/activate  # 或 conda activate whisper-amd

# 进入项目目录
cd ~/Auto-Subtitle-on-Generative-AI

# 启动 AMD 优化版本
python start_amd.py
```

### 访问应用
- **实时转录**: http://localhost:5001/realtime.html
- **文件处理**: http://localhost:5001/app.html

---

🎮 **AMD 笔记本用户专享的 AI 字幕生成器部署完成！** 🚀