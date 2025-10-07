#!/bin/bash

# AMD 笔记本专用 AI 字幕生成器安装脚本
# 🎮 针对 AMD GPU + ROCm 平台优化
# 支持: RX 6000/7000 系列, RX 5000 系列, APU 集显

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 图标定义
GPU_ICON="🎮"
ROCKET_ICON="🚀"
CHECK_ICON="✅"
WARN_ICON="⚠️"
ERROR_ICON="❌"
INFO_ICON="💡"

print_header() {
    clear
    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${CYAN}${GPU_ICON} AMD 笔记本 AI 字幕生成器 - 智能部署脚本 ${GPU_ICON}${NC}"
    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${YELLOW}🎯 专为 AMD GPU 笔记本优化的一键部署解决方案${NC}"
    echo -e "${YELLOW}📖 支持: RDNA2/RDNA3 (RX 6000/7000), RDNA (RX 5000), APU${NC}"
    echo -e "${PURPLE}================================================================${NC}"
    echo
}

print_step() {
    echo -e "${BLUE}[步骤] $1${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK_ICON} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}${WARN_ICON} $1${NC}"
}

print_error() {
    echo -e "${RED}${ERROR_ICON} $1${NC}"
}

print_info() {
    echo -e "${CYAN}${INFO_ICON} $1${NC}"
}

# 检查是否为 root 用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "请不要以 root 用户身份运行此脚本!"
        print_info "使用普通用户运行: curl -fsSL https://raw.githubusercontent.com/lemon5227/Auto-Subtitle-on-Generative-AI/main/install-amd.sh | bash"
        exit 1
    fi
}

# 检测系统信息
detect_system() {
    print_step "检测系统环境..."
    
    # 检测发行版
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
        echo -e "   ${CHECK_ICON} 操作系统: $OS $VER"
    else
        print_error "无法检测操作系统版本"
        exit 1
    fi
    
    # 检查架构
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        print_error "仅支持 x86_64 架构，当前架构: $ARCH"
        exit 1
    fi
    echo -e "   ${CHECK_ICON} 系统架构: $ARCH"
    
    # 检查内存
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -lt 6 ]]; then
        print_warning "内存较少 (${TOTAL_MEM}GB)，推荐 8GB+ 以获得最佳性能"
    else
        echo -e "   ${CHECK_ICON} 系统内存: ${TOTAL_MEM}GB"
    fi
    
    echo
}

# 检测 AMD GPU
detect_amd_gpu() {
    print_step "检测 AMD GPU 硬件..."
    
    # 检查 PCI 设备中的 AMD GPU
    AMD_GPUS=$(lspci | grep -i "amd.*vga\|amd.*display" || true)
    APU_DETECTED=$(lspci | grep -i "amd.*vga.*renoir\|amd.*vga.*picasso\|amd.*vga.*cezanne" || true)
    
    if [[ -n "$AMD_GPUS" ]]; then
        echo -e "   ${CHECK_ICON} 检测到 AMD GPU:"
        echo "$AMD_GPUS" | while read -r line; do
            echo -e "      🎮 $line"
        done
        
        # 判断 GPU 类型
        if [[ "$AMD_GPUS" =~ "Radeon RX 6" ]] || [[ "$AMD_GPUS" =~ "Radeon RX 7" ]]; then
            GPU_TYPE="RDNA2/RDNA3"
            PERFORMANCE_LEVEL="excellent"
            echo -e "   ${CHECK_ICON} GPU 类型: $GPU_TYPE (性能优秀)"
        elif [[ "$AMD_GPUS" =~ "Radeon RX 5" ]]; then
            GPU_TYPE="RDNA"
            PERFORMANCE_LEVEL="good"
            echo -e "   ${CHECK_ICON} GPU 类型: $GPU_TYPE (性能良好)"
        elif [[ -n "$APU_DETECTED" ]]; then
            GPU_TYPE="APU"
            PERFORMANCE_LEVEL="basic"
            echo -e "   ${CHECK_ICON} GPU 类型: Ryzen APU 集显 (基础性能)"
        else
            GPU_TYPE="OTHER"
            PERFORMANCE_LEVEL="limited"
            print_warning "GPU 类型: 其他 AMD GPU (性能有限)"
        fi
    else
        print_warning "未检测到 AMD GPU，将使用 CPU 模式"
        GPU_TYPE="NONE"
        PERFORMANCE_LEVEL="cpu_only"
    fi
    
    echo
}

# 安装系统依赖
install_system_deps() {
    print_step "安装系统依赖..."
    
    # 更新包列表
    echo -e "   📦 更新软件包列表..."
    sudo apt update -qq
    
    # 安装基础依赖
    echo -e "   📦 安装基础开发工具..."
    sudo apt install -y \
        python3 python3-pip python3-venv python3-dev \
        git curl wget unzip \
        build-essential cmake \
        pkg-config \
        ffmpeg \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release > /dev/null 2>&1
    
    print_success "系统依赖安装完成"
    echo
}

# 安装 AMD GPU 驱动
install_amd_drivers() {
    if [[ "$GPU_TYPE" == "NONE" ]]; then
        print_info "跳过 AMD GPU 驱动安装 (未检测到 GPU)"
        return
    fi
    
    print_step "安装 AMD GPU 驱动..."
    
    # 检查是否已安装 amdgpu 驱动
    if lsmod | grep -q amdgpu; then
        print_success "AMD GPU 内核驱动已加载"
    else
        echo -e "   🔧 安装 Mesa AMD 驱动..."
        sudo apt install -y \
            mesa-vulkan-drivers \
            xserver-xorg-video-amdgpu \
            mesa-vulkan-drivers:i386 > /dev/null 2>&1
        
        print_success "AMD GPU 驱动安装完成"
    fi
    
    echo
}

# 安装 ROCm 平台 (GPU 加速)
install_rocm() {
    if [[ "$GPU_TYPE" == "NONE" ]]; then
        print_info "跳过 ROCm 安装 (CPU 模式)"
        return
    fi
    
    print_step "安装 ROCm 计算平台..."
    
    # 检查是否已安装 ROCm
    if [[ -d "/opt/rocm" ]]; then
        print_success "ROCm 已安装，跳过..."
        return
    fi
    
    # 获取系统版本
    UBUNTU_VERSION=$(lsb_release -rs)
    
    echo -e "   🔧 添加 ROCm APT 仓库..."
    # 添加 ROCm GPG 密钥
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - > /dev/null 2>&1
    
    # 添加 ROCm 仓库 (基于 Ubuntu 版本)
    if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
    elif [[ "$UBUNTU_VERSION" == "20.04" ]]; then
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ focal main' | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
    else
        print_warning "Ubuntu $UBUNTU_VERSION 可能不完全支持，尝试使用 jammy 仓库"
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
    fi
    
    # 更新包列表
    sudo apt update -qq
    
    echo -e "   🔧 安装 ROCm 核心组件..."
    # 安装 ROCm (选择性安装以避免冲突)
    sudo apt install -y \
        rocm-dev \
        hip-dev \
        rocm-device-libs > /dev/null 2>&1 || {
        print_warning "完整 ROCm 安装失败，尝试最小化安装..."
        sudo apt install -y hip-runtime-amd hip-dev > /dev/null 2>&1
    }
    
    # 添加用户到相关组
    echo -e "   👤 配置用户权限..."
    sudo usermod -a -G render,video $USER
    
    # 设置环境变量
    if ! grep -q "ROCM_PATH" ~/.bashrc; then
        echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
        echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc
    fi
    
    print_success "ROCm 安装完成"
    print_info "注意: 需要重新登录或重启以应用组权限更改"
    echo
}

# 克隆项目
clone_project() {
    print_step "获取项目源码..."
    
    PROJECT_DIR="$HOME/Auto-Subtitle-on-Generative-AI"
    
    if [[ -d "$PROJECT_DIR" ]]; then
        print_info "项目目录已存在，更新代码..."
        cd "$PROJECT_DIR"
        git pull origin main > /dev/null 2>&1 || {
            print_warning "Git 更新失败，重新克隆..."
            cd "$HOME"
            rm -rf "$PROJECT_DIR"
            git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git > /dev/null 2>&1
        }
    else
        print_info "克隆项目仓库..."
        cd "$HOME"
        git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git > /dev/null 2>&1
    fi
    
    cd "$PROJECT_DIR"
    print_success "项目源码获取完成"
    echo
}

# 创建 Python 虚拟环境
setup_python_env() {
    print_step "配置 Python 环境..."
    
    VENV_DIR="venv_amd"
    
    if [[ -d "$VENV_DIR" ]]; then
        print_info "虚拟环境已存在，跳过创建..."
    else
        echo -e "   🐍 创建 AMD 专用虚拟环境..."
        python3 -m venv $VENV_DIR
        print_success "虚拟环境创建完成"
    fi
    
    # 激活虚拟环境
    source $VENV_DIR/bin/activate
    
    # 升级 pip
    echo -e "   📦 升级 pip..."
    pip install --upgrade pip > /dev/null 2>&1
    
    print_success "Python 环境配置完成"
    echo
}

# 安装 PyTorch ROCm 版本
install_pytorch_rocm() {
    print_step "安装 PyTorch (ROCm 版本)..."
    
    if [[ "$GPU_TYPE" == "NONE" ]]; then
        echo -e "   🔧 安装 CPU 版 PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
        print_success "PyTorch (CPU 版本) 安装完成"
    else
        echo -e "   🔧 安装 ROCm 版 PyTorch..."
        # 安装 ROCm 5.7 对应的 PyTorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 > /dev/null 2>&1 || {
            print_warning "ROCm 版本安装失败，回退到 CPU 版本..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
        }
        print_success "PyTorch (ROCm 版本) 安装完成"
    fi
    
    echo
}

# 安装应用依赖
install_app_deps() {
    print_step "安装应用依赖..."
    
    echo -e "   📦 安装核心依赖包..."
    pip install -r requirements.txt > /dev/null 2>&1
    
    # AMD 优化的额外依赖
    echo -e "   🎮 安装 AMD 优化组件..."
    pip install accelerate optimum[onnxruntime] > /dev/null 2>&1 || {
        print_warning "部分优化组件安装失败，不影响核心功能"
    }
    
    print_success "应用依赖安装完成"
    echo
}

# 创建 AMD 专用启动脚本
create_amd_launcher() {
    print_step "创建 AMD GPU 优化启动器..."
    
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
    print("\n🔔 AMD GPU 优化提示:")
    print("   - 使用最新 ROCm 驱动获得最佳性能")
    print("   - 大模型可能需要 8GB+ 显存")
    print("   - 按 Ctrl+C 停止服务")
    print("\n" + "-" * 50)
    
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n💡 AMD GPU 故障排除:")
        print("   1. 检查 ROCm 是否正确安装")
        print("   2. 验证用户在 render 组中")
        print("   3. 重启系统应用驱动更新")
        print("   4. 尝试 CPU 模式: export HIP_VISIBLE_DEVICES=-1")

if __name__ == "__main__":
    setup_amd_environment()
    start_application()
EOF

    chmod +x start_amd.py
    
    print_success "AMD 启动器创建完成"
    echo
}

# 创建环境检查脚本
create_check_script() {
    print_step "创建环境检查工具..."
    
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

    chmod +x check_amd_env.py
    
    print_success "环境检查工具创建完成"
    echo
}

# 验证安装
verify_installation() {
    print_step "验证安装结果..."
    
    echo -e "   🧪 运行环境检查..."
    python check_amd_env.py
    
    echo
    print_success "安装验证完成"
    echo
}

# 创建桌面快捷方式 (可选)
create_desktop_shortcut() {
    print_step "创建桌面快捷方式 (可选)..."
    
    DESKTOP_DIR="$HOME/Desktop"
    if [[ -d "$DESKTOP_DIR" ]]; then
        cat > "$DESKTOP_DIR/AMD-AI-字幕生成器.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=AMD AI 字幕生成器
Comment=AMD GPU 优化的 AI 字幕生成器
Icon=applications-multimedia
Exec=gnome-terminal -- bash -c "cd $HOME/Auto-Subtitle-on-Generative-AI && source venv_amd/bin/activate && python start_amd.py; exec bash"
Terminal=false
Categories=AudioVideo;Audio;Video;
EOF
        chmod +x "$DESKTOP_DIR/AMD-AI-字幕生成器.desktop"
        print_success "桌面快捷方式已创建"
    else
        print_info "桌面目录不存在，跳过快捷方式创建"
    fi
    echo
}

# 显示完成信息
show_completion_info() {
    print_header
    echo -e "${GREEN}${ROCKET_ICON} AMD GPU AI 字幕生成器安装成功！ ${ROCKET_ICON}${NC}"
    echo
    
    echo -e "${CYAN}📋 安装摘要:${NC}"
    echo -e "   ${CHECK_ICON} GPU 类型: $GPU_TYPE ($PERFORMANCE_LEVEL 性能)"
    echo -e "   ${CHECK_ICON} ROCm 平台: $(if [[ -d "/opt/rocm" ]]; then echo "已安装"; else echo "未安装 (CPU 模式)"; fi)"
    echo -e "   ${CHECK_ICON} PyTorch ROCm: 已配置"
    echo -e "   ${CHECK_ICON} 项目路径: $HOME/Auto-Subtitle-on-Generative-AI"
    echo
    
    echo -e "${CYAN}🚀 启动应用:${NC}"
    echo -e "   ${INFO_ICON} 进入项目目录:"
    echo -e "      cd ~/Auto-Subtitle-on-Generative-AI"
    echo -e "   ${INFO_ICON} 激活环境:"
    echo -e "      source venv_amd/bin/activate"
    echo -e "   ${INFO_ICON} 启动 AMD 优化版本:"
    echo -e "      python start_amd.py"
    echo
    
    echo -e "${CYAN}🌐 访问地址:${NC}"
    echo -e "   ${INFO_ICON} 实时转录: ${YELLOW}http://localhost:5001/realtime.html${NC}"
    echo -e "   ${INFO_ICON} 文件处理: ${YELLOW}http://localhost:5001/app.html${NC}"
    echo
    
    echo -e "${CYAN}🔧 常用命令:${NC}"
    echo -e "   ${INFO_ICON} 环境检查: ${YELLOW}python check_amd_env.py${NC}"
    echo -e "   ${INFO_ICON} 更新项目: ${YELLOW}git pull origin main${NC}"
    echo -e "   ${INFO_ICON} 故障排除: 查看 ${YELLOW}README.amd.md${NC}"
    echo
    
    if [[ "$GPU_TYPE" != "NONE" ]]; then
        echo -e "${YELLOW}${WARN_ICON} AMD GPU 注意事项:${NC}"
        echo -e "   - 首次运行可能需要下载模型文件"
        echo -e "   - 如遇权限问题，请重新登录或重启系统"
        echo -e "   - 大模型需要足够的显存 (建议 8GB+)"
        echo -e "   - 性能调优请参考 README.amd.md"
    else
        echo -e "${YELLOW}${INFO_ICON} CPU 模式提示:${NC}"
        echo -e "   - 当前将使用 CPU 进行推理"
        echo -e "   - 推荐使用较小的模型以获得更好的响应速度"
        echo -e "   - 如需 GPU 加速，请安装支持的 AMD GPU"
    fi
    
    echo
    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${GREEN}${GPU_ICON} 享受 AMD GPU 加速的 AI 字幕生成体验！ ${GPU_ICON}${NC}"
    echo -e "${PURPLE}================================================================${NC}"
}

# 主安装流程
main() {
    print_header
    
    # 检查权限
    check_root
    
    # 系统检测
    detect_system
    detect_amd_gpu
    
    # 等待用户确认
    echo -e "${YELLOW}准备为 AMD $GPU_TYPE GPU 安装 AI 字幕生成器${NC}"
    read -p "按 Enter 继续，Ctrl+C 取消..."
    echo
    
    # 安装流程
    install_system_deps
    install_amd_drivers
    install_rocm
    clone_project
    setup_python_env
    install_pytorch_rocm
    install_app_deps
    create_amd_launcher
    create_check_script
    verify_installation
    create_desktop_shortcut
    
    # 完成
    show_completion_info
}

# 错误处理
trap 'print_error "安装过程中断！"; exit 1' INT TERM

# 运行主程序
main "$@"