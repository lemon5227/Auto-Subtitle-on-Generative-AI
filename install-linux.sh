#!/bin/bash
# =============================================================================
# Auto Subtitle Generator - Linux 一键安装脚本
# 🚀 丝滑部署，支持所有主流Linux发行版
# =============================================================================

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 检测Linux发行版
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    elif [[ -f /etc/redhat-release ]]; then
        DISTRO="centos"
    elif [[ -f /etc/debian_version ]]; then
        DISTRO="debian"
    else
        DISTRO="unknown"
    fi
    
    log_info "检测到系统: $DISTRO $VERSION"
}

# 安装系统依赖
install_system_deps() {
    log_info "安装系统依赖..."
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv git curl ffmpeg
            ;;
        centos|rhel)
            sudo yum install -y epel-release
            sudo yum install -y python3 python3-pip git curl ffmpeg
            ;;
        fedora)
            sudo dnf install -y python3 python3-pip git curl ffmpeg
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm python python-pip git curl ffmpeg
            ;;
        *)
            log_warning "未识别的发行版，请手动安装: python3 python3-pip git curl ffmpeg"
            return 1
            ;;
    esac
    
    log_success "系统依赖安装完成"
}

# 检查Python版本
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        log_info "Python版本: $PYTHON_VERSION"
        
        # 检查版本是否满足要求
        if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
            log_success "Python版本满足要求"
            return 0
        else
            log_error "需要Python 3.8+，当前版本: $PYTHON_VERSION"
            return 1
        fi
    else
        log_error "Python3未找到"
        return 1
    fi
}

# 克隆或更新仓库
setup_repo() {
    if [[ -d "Auto-Subtitle-on-Generative-AI" ]]; then
        log_info "项目目录已存在，更新代码..."
        cd Auto-Subtitle-on-Generative-AI
        git pull origin main || log_warning "代码更新失败，继续使用本地版本"
    else
        log_info "克隆项目仓库..."
        git clone https://github.com/lemon5227/Auto-Subtitle-on-Generative-AI.git
        cd Auto-Subtitle-on-Generative-AI
    fi
    
    log_success "项目代码准备完成"
}

# 创建虚拟环境
setup_venv() {
    log_info "创建Python虚拟环境..."
    
    if [[ -d "venv" ]]; then
        log_info "虚拟环境已存在，跳过创建"
    else
        python3 -m venv venv
        log_success "虚拟环境创建完成"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    log_success "虚拟环境已激活"
}

# 安装Python依赖
install_python_deps() {
    log_info "安装Python依赖包..."
    
    # 升级pip
    pip install --upgrade pip
    
    # 检测GPU支持
    if command -v nvidia-smi &> /dev/null; then
        log_info "检测到NVIDIA GPU，安装CUDA版本PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "安装CPU版本PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # 安装其他依赖
    pip install -r requirements.txt
    
    log_success "Python依赖安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 检查关键组件
    if ! command -v ffmpeg &> /dev/null; then
        log_error "ffmpeg未正确安装"
        return 1
    fi
    
    if ! python -c "import torch, flask, whisper" &> /dev/null; then
        log_error "Python依赖未正确安装"
        return 1
    fi
    
    log_success "安装验证通过"
}

# 启动应用
start_app() {
    log_info "启动AI字幕生成器..."
    log_info "请在浏览器中访问："
    echo -e "  ${GREEN}📺 实时转录:${NC} http://127.0.0.1:5001/realtime.html"
    echo -e "  ${GREEN}🎬 文件处理:${NC} http://127.0.0.1:5001/app.html"
    echo ""
    echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"
    echo ""
    
    python app.py
}

# 主函数
main() {
    echo "======================================================"
    echo "🚀 Auto Subtitle Generator - Linux一键部署脚本"
    echo "🐧 支持 Ubuntu/Debian/CentOS/Fedora/Arch Linux"
    echo "======================================================"
    
    # 检查是否为root用户
    if [[ $EUID -eq 0 ]]; then
        log_warning "检测到root用户，建议使用普通用户运行"
    fi
    
    # 检测系统
    detect_distro
    
    # 安装系统依赖
    if ! command -v python3 &> /dev/null || ! command -v ffmpeg &> /dev/null; then
        install_system_deps
    else
        log_success "系统依赖已安装"
    fi
    
    # 检查Python
    if ! check_python; then
        log_error "Python环境检查失败"
        exit 1
    fi
    
    # 设置项目
    setup_repo
    
    # 设置虚拟环境
    setup_venv
    
    # 安装Python依赖
    install_python_deps
    
    # 验证安装
    verify_installation
    
    # 启动应用
    echo ""
    log_success "🎉 安装完成！"
    echo ""
    read -p "是否现在启动应用? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_app
    else
        echo ""
        echo "手动启动命令:"
        echo "  cd Auto-Subtitle-on-Generative-AI"
        echo "  source venv/bin/activate"
        echo "  python app.py"
        echo ""
        log_success "感谢使用!"
    fi
}

# 错误处理
trap 'log_error "安装过程中发生错误，请检查上方日志"; exit 1' ERR

# 运行主函数
main "$@"