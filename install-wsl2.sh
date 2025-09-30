#!/bin/bash
# =============================================================================
# Auto Subtitle Generator - WSL2 专用一键安装脚本  
# 🚀 针对 Windows WSL2 环境优化的丝滑部署
# =============================================================================

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_wsl2() {
    echo -e "${PURPLE}[WSL2]${NC} $1"
}

# WSL2 环境检测
check_wsl2_environment() {
    log_info "检测 WSL2 环境..."
    
    # 检查是否在 WSL2 中
    if [[ ! -f /proc/version ]] || ! grep -qi microsoft /proc/version; then
        log_error "不在 WSL 环境中运行"
        exit 1
    fi
    
    if grep -qi "WSL2" /proc/version || [[ -d "/sys/fs/cgroup/unified" ]]; then
        log_success "确认运行在 WSL2 环境中"
        WSL_VERSION=2
    else
        log_warning "可能在 WSL1 环境中，建议升级到 WSL2"
        WSL_VERSION=1
    fi
    
    # 显示发行版信息
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log_info "Linux 发行版: $PRETTY_NAME"
        
        # 检查推荐的发行版
        case $ID in
            ubuntu)
                if [[ "$VERSION_ID" < "20.04" ]]; then
                    log_warning "建议使用 Ubuntu 20.04 或更新版本"
                fi
                ;;
            debian)
                if [[ "$VERSION_ID" < "11" ]]; then
                    log_warning "建议使用 Debian 11 或更新版本"
                fi
                ;;
        esac
    fi
}

# 检查 Windows 端 NVIDIA 支持
check_windows_nvidia() {
    log_info "检查 Windows 端 NVIDIA 支持..."
    
    # 检查 WSL 库文件
    if [[ -d "/usr/lib/wsl/lib" ]] && ls /usr/lib/wsl/lib/libcuda.so* >/dev/null 2>&1; then
        log_success "检测到 WSL GPU 库文件"
        WSL_GPU_SUPPORT=true
        
        # 列出可用的 CUDA 库
        log_info "可用的 CUDA 库:"
        ls -la /usr/lib/wsl/lib/ | grep -E "(cuda|nv)" || true
    else
        log_warning "未检测到 WSL GPU 支持库"
        log_info "GPU 加速将不可用，将使用 CPU 模式"
        WSL_GPU_SUPPORT=false
    fi
}

# WSL2 系统优化
optimize_wsl2_system() {
    log_wsl2 "应用 WSL2 系统优化..."
    
    # 设置环境变量优化
    cat >> ~/.bashrc << 'EOF'

# WSL2 AI 优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# HuggingFace 国内镜像 (可选)
# export HF_ENDPOINT=https://hf-mirror.com
EOF

    log_success "WSL2 环境变量已配置"
}

# 检查和安装系统依赖
install_system_deps() {
    log_info "更新包管理器..."
    sudo apt update
    
    log_info "安装系统依赖..."
    sudo apt install -y \
        python3 python3-pip python3-venv python3-dev \
        git curl wget unzip \
        ffmpeg \
        build-essential \
        software-properties-common \
        ca-certificates \
        apt-transport-https
    
    log_success "系统依赖安装完成"
}

# 安装 Conda (针对 WSL2 优化)
install_conda() {
    if command -v conda &> /dev/null; then
        log_success "Conda 已安装"
        return 0
    fi
    
    log_info "安装 Miniconda (WSL2 优化版)..."
    
    # 下载 Miniconda
    CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    wget -q "$CONDA_URL" -O miniconda.sh
    
    # 静默安装
    bash miniconda.sh -b -p "$HOME/miniconda"
    
    # 添加到 PATH
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # 初始化 conda
    "$HOME/miniconda/bin/conda" init bash
    
    # 清理
    rm miniconda.sh
    
    log_success "Miniconda 安装完成"
}

# 创建 Python 环境
setup_python_env() {
    log_info "创建 Python 环境..."
    
    # 确保 conda 可用
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # 创建环境
    conda create -n whisper-app python=3.11 -y
    
    log_success "Python 环境创建完成"
}

# 克隆或更新项目
setup_project() {
    log_info "设置项目代码..."
    
    PROJECT_DIR="$HOME/Auto-Subtitle-on-Generative-AI"
    
    if [[ -d "$PROJECT_DIR" ]]; then
        log_info "项目目录已存在，更新代码..."
        cd "$PROJECT_DIR"
        git pull origin main || log_warning "代码更新失败，使用本地版本"
    else
        log_info "克隆项目仓库..."
        git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi
    
    log_success "项目代码已准备完成"
}

# 安装 Python 依赖 (WSL2 特化)
install_python_deps() {
    log_info "激活 Python 环境并安装依赖..."
    
    # 激活环境
    source "$HOME/miniconda/bin/activate" whisper-app
    
    # 升级基础工具
    pip install --upgrade pip setuptools wheel
    
    # WSL2 专用 PyTorch 安装
    if [[ "$WSL_GPU_SUPPORT" == "true" ]]; then
        log_info "安装 CUDA 版本 PyTorch (WSL2 优化)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "安装 CPU 版本 PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # 安装应用依赖
    log_info "安装应用依赖包..."
    pip install -r requirements.txt
    
    log_success "Python 依赖安装完成"
}

# 创建 WSL2 专用启动器
create_wsl2_launcher() {
    log_info "创建 WSL2 专用启动器..."
    
    cat > start_wsl2.py << 'EOF'
#!/usr/bin/env python3
"""
WSL2 专用启动脚本 - AI 字幕生成器
🚀 针对 WSL2 环境优化，解决常见兼容性问题
"""
import os
import sys
import subprocess
import platform

def print_wsl2_info():
    print("=" * 60)
    print("🔧 WSL2 优化启动器 - AI 字幕生成器")
    print("🚀 针对 Windows WSL2 环境优化")
    print("=" * 60)
    
    # 显示环境信息
    print(f"🐧 Linux 发行版: {platform.platform()}")
    print(f"🐍 Python 版本: {sys.version.split()[0]}")
    
    # WSL2 优化设置
    optimizations = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'CUDA_LAUNCH_BLOCKING': '1',
        'OMP_NUM_THREADS': '4',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    print("\n🔧 WSL2 优化配置:")
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"   {key}={value}")

def check_gpu_support():
    """检查 GPU 支持"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n🎮 GPU 支持: {'✅ 可用' if cuda_available else '❌ 不可用 (将使用 CPU)'}")
        
        if cuda_available:
            print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
        
        return cuda_available
    except Exception as e:
        print(f"\n⚠️  GPU 检查失败: {e}")
        return False

def start_app():
    """启动应用"""
    print("\n🌐 访问地址:")
    print("   📺 实时转录: http://localhost:5001/realtime.html")
    print("   🎬 文件处理: http://localhost:5001/app.html")
    print("\n🔔 提示:")
    print("   - 在 Windows 浏览器中打开上述地址")
    print("   - 按 Ctrl+C 停止服务")
    print("   - 如遇 CUDA 错误，服务会自动回退到 CPU 模式")
    print("\n" + "-" * 60)
    
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止，谢谢使用!")
    except FileNotFoundError:
        print("\n❌ app.py 文件未找到")
        print("📁 请确保在项目根目录运行此脚本")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n💡 故障排除建议:")
        print("   1. 检查端口 5001 是否被占用")
        print("   2. 尝试重新激活 conda 环境")
        print("   3. 如果是 GPU 错误，可以强制使用 CPU:")
        print("      export CUDA_VISIBLE_DEVICES=''")
        print("      python app.py")

if __name__ == "__main__":
    print_wsl2_info()
    check_gpu_support()
    start_app()
EOF

    chmod +x start_wsl2.py
    log_success "WSL2 启动器已创建: start_wsl2.py"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 激活环境进行验证
    source "$HOME/miniconda/bin/activate" whisper-app
    
    # 检查关键组件
    python -c "
import sys
try:
    import torch, flask, whisper
    print('✅ 核心依赖检查通过')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA 可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ 依赖检查失败: {e}')
    sys.exit(1)
" || {
        log_error "Python 依赖验证失败"
        return 1
    }
    
    # 检查 ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        log_error "ffmpeg 未正确安装"
        return 1
    fi
    
    log_success "安装验证通过"
}

# 显示完成信息
show_completion_info() {
    echo ""
    echo "🎉" "=" * 50
    echo "🎊 WSL2 环境 AI 字幕生成器安装完成!"
    echo "=" * 55
    echo ""
    echo "📋 快速启动指南:"
    echo "   1️⃣ 激活环境: conda activate whisper-app"
    echo "   2️⃣ 进入目录: cd ~/Auto-Subtitle-on-Generative-AI"
    echo "   3️⃣ 启动服务: python start_wsl2.py"
    echo ""
    echo "🌐 或者直接运行 WSL2 优化启动器:"
    echo "   cd ~/Auto-Subtitle-on-Generative-AI && conda activate whisper-app && python start_wsl2.py"
    echo ""
    echo "🔗 访问地址 (在 Windows 浏览器中):"
    echo "   📺 实时转录: http://localhost:5001/realtime.html"
    echo "   🎬 文件处理: http://localhost:5001/app.html"
    echo ""
    echo "💡 小提示:"
    echo "   - WSL2 环境已优化，支持 GPU 加速 (如果可用)"
    echo "   - 项目位于: ~/Auto-Subtitle-on-Generative-AI"
    echo "   - 遇到问题查看: README.wsl2.md"
    echo ""
    echo "🎯 性能建议:"
    echo "   - 确保 Windows 分配足够内存给 WSL2 (建议 8GB+)"
    echo "   - 使用 localhost 而不是 127.0.0.1 访问服务"
    echo ""
}

# 主函数
main() {
    echo "======================================================"
    echo "🚀 Auto Subtitle Generator - WSL2 一键部署脚本"
    echo "🔧 专为 Windows WSL2 环境优化"
    echo "======================================================"
    
    # 检查 WSL2 环境
    check_wsl2_environment
    
    # 检查 GPU 支持
    check_windows_nvidia
    
    # 系统优化
    optimize_wsl2_system
    
    # 安装系统依赖
    install_system_deps
    
    # 安装 Conda
    install_conda
    
    # 设置项目
    setup_project
    
    # 创建 Python 环境
    setup_python_env
    
    # 安装 Python 依赖
    install_python_deps
    
    # 创建启动器
    create_wsl2_launcher
    
    # 验证安装
    if verify_installation; then
        show_completion_info
    else
        log_error "安装验证失败，请检查错误信息"
        exit 1
    fi
    
    # 询问是否立即启动
    echo ""
    read -p "是否现在启动应用? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        log_info "启动 AI 字幕生成器..."
        cd ~/Auto-Subtitle-on-Generative-AI
        source "$HOME/miniconda/bin/activate" whisper-app
        python start_wsl2.py
    else
        echo ""
        log_success "安装完成! 使用上述命令手动启动服务"
    fi
}

# 错误处理
trap 'log_error "安装过程中发生错误 (${BASH_SOURCE}:${LINENO})，请检查上方日志"; exit 1' ERR

# 运行主函数
main "$@"