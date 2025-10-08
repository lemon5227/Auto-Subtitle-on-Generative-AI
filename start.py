#!/usr/bin/env python3
"""
智能启动器 - AI 字幕生成器
🚀 自动检测并优化 NVIDIA / AMD / Apple Silicon / CPU 环境
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_header():
    """打印启动头部"""
    print("🚀 AI 字幕生成器 - 智能启动器")
    print("=" * 60)
    print("🎯 自动检测最佳计算设备并优化性能配置")
    print("� 支持: NVIDIA CUDA / AMD ROCm / Apple MPS / CPU")
    print("=" * 60)

def check_dependencies():
    """检查基础依赖"""
    try:
        import torch
        import whisper
        import flask
        logger.info("✅ 核心依赖检查通过")
        return True
    except ImportError as e:
        logger.error(f"❌ 缺少依赖: {e}")
        logger.error("💡 请运行: pip install -r requirements.txt")
        return False

def setup_environment():
    """设置环境和导入GPU检测"""
    try:
        from gpu_detector import GPUDetector, create_device_environment
        
        # 创建检测器
        detector = GPUDetector()
        
        # 打印检测报告
        print("\n🔍 GPU 环境检测:")
        print("-" * 40)
        print(f"📊 {detector.get_device_summary()}")
        
        # 应用环境变量优化
        device_env = create_device_environment()
        for key, value in device_env.items():
            os.environ[key] = value
            
        # 显示关键优化设置
        device_type = detector.device_info['gpu_type']
        if device_type == 'nvidia':
            print("🟢 NVIDIA CUDA 优化已启用")
        elif device_type == 'amd':
            print("🔴 AMD ROCm 优化已启用")
        elif device_type == 'apple':
            print("🟡 Apple MPS 优化已启用")
        else:
            print("🔵 CPU 多线程优化已启用")
            
        # 显示优化建议
        tips = detector.get_optimization_tips()
        if tips:
            print(f"💡 {tips[0]}")  # 显示第一个建议
            
        return detector
        
    except ImportError:
        logger.warning("⚠️ GPU 检测模块不可用，使用默认配置")
        return None
    
    print(f"🖥️  系统: {system} {platform.release()}")
    print(f"🔧 架构: {machine}")
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return system, machine

def auto_install_system_deps():
    """自动安装系统依赖（Linux平台）"""
    system, _ = check_system()
    
    if system != "Linux":
        return True
    
    print("\n� 检测Linux发行版并安装系统依赖...")
    
    # 检测Linux发行版
    try:
        with open('/etc/os-release', 'r') as f:
            os_info = f.read()
        
        if 'Ubuntu' in os_info or 'Debian' in os_info:
            distro = 'ubuntu'
        elif 'CentOS' in os_info or 'Red Hat' in os_info or 'rhel' in os_info:
            distro = 'centos'
        elif 'Fedora' in os_info:
            distro = 'fedora'
        else:
            distro = 'unknown'
            
        print(f"🐧 检测到发行版: {distro}")
        
        # 自动安装ffmpeg
        if not shutil.which('ffmpeg'):
            print("📦 自动安装 ffmpeg...")
            if distro == 'ubuntu':
                result = subprocess.run(['sudo', 'apt', 'update'], capture_output=True)
                if result.returncode == 0:
                    subprocess.run(['sudo', 'apt', 'install', '-y', 'ffmpeg'], check=True)
                    print("✅ ffmpeg 安装完成")
            elif distro == 'fedora':
                subprocess.run(['sudo', 'dnf', 'install', '-y', 'ffmpeg'], check=True)
                print("✅ ffmpeg 安装完成")
            elif distro == 'centos':
                subprocess.run(['sudo', 'yum', 'install', '-y', 'epel-release'], check=True)
                subprocess.run(['sudo', 'yum', 'install', '-y', 'ffmpeg'], check=True)
                print("✅ ffmpeg 安装完成")
        else:
            print("✅ ffmpeg 已安装")
            
    except Exception as e:
        print(f"⚠️  自动安装失败: {e}")
        print("📝 请手动安装 ffmpeg")
        return False
        
    return True

def setup_python_env():
    """设置Python环境"""
    print("\n🐍 配置Python环境...")
    
    # 检查是否在虚拟环境中
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, 'real_prefix')
    
    if not in_venv:
        print("💡 建议创建虚拟环境，是否自动创建? (y/n): ", end="")
        try:
            choice = input().lower().strip()
            if choice in ['y', 'yes', '']:
                print("📦 创建虚拟环境...")
                env_path = Path("venv")
                if not env_path.exists():
                    subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
                    print("✅ 虚拟环境创建完成")
                    
                # 激活虚拟环境的说明
                system, _ = check_system()
                if system == "Windows":
                    activate_cmd = ".\\venv\\Scripts\\activate"
                else:
                    activate_cmd = "source venv/bin/activate"
                    
                print(f"🔔 请运行以下命令激活环境:")
                print(f"   {activate_cmd}")
                print("   然后重新运行: python start.py")
                return False
        except KeyboardInterrupt:
            print("\n⏭️  跳过虚拟环境创建")
    
    return True

def check_and_install_dependencies():
    """检查并自动安装依赖项"""
    print("\n📦 检查Python依赖项...")
    
    # 检查requirements.txt
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt 未找到")
        return False
    
    try:
        # 检查torch
        import torch
        print(f"✅ PyTorch {torch.__version__} 已安装")
        
        # 检查加速支持
        if torch.cuda.is_available():
            print(f"🚀 CUDA 可用: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🚀 MPS (Apple Silicon GPU) 可用")
        else:
            print("⚠️  使用 CPU 模式")
            
    except ImportError:
        print("📥 安装 PyTorch...")
        system, machine = check_system()
        
        # 智能选择PyTorch版本
        if system == "Darwin" and machine == "arm64":  # Apple Silicon
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'], check=True)
        elif system == "Linux":
            # 检测CUDA
            if shutil.which('nvidia-smi'):
                print("🎮 检测到NVIDIA GPU，安装CUDA版本...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'], check=True)
            else:
                print("💻 安装CPU版本...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
        else:
            # Windows或其他
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'], check=True)
    
    # 安装其他依赖
    print("📦 安装应用依赖...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    print("✅ 所有依赖安装完成")
    
    return True

def check_system_deps():
    """检查系统依赖"""
    print("\n🔍 检查系统依赖...")
    
    # 检查 ffmpeg
    if shutil.which('ffmpeg'):
        print("✅ ffmpeg 已安装")
        return True
    else:
        print("❌ ffmpeg 未找到")
        system, _ = check_system()
        
        # Linux自动安装
        if system == "Linux":
            print("🚀 尝试自动安装...")
            return auto_install_system_deps()
        elif system == "Darwin":  # macOS
            print("📝 安装命令: brew install ffmpeg")
        elif system == "Windows":
            print("📝 请从 https://ffmpeg.org/ 下载并添加到 PATH")
        
        return False

def main():
    """智能启动主函数"""
    print("🎤 Auto Subtitle Generator - 智能启动器")
    print("🚀 支持 Linux 丝滑一键部署 / macOS Apple Silicon / Windows")
    print("=" * 60)
    
    system, machine = check_system()
    
    # 1. 检查系统依赖
    if not check_system_deps():
        print("\n❌ 系统依赖检查失败")
        if system != "Linux":
            print("📝 请手动安装依赖后重试")
            sys.exit(1)
    
    # 2. 设置Python环境
    if not setup_python_env():
        sys.exit(0)  # 需要激活虚拟环境
    
    # 3. 安装Python依赖
    try:
        if not check_and_install_dependencies():
            print("\n❌ Python依赖安装失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 依赖安装错误: {e}")
        sys.exit(1)
    
    # 4. 启动应用
    print("\n🚀 启动Web应用...")
    print("🌐 访问地址:")
    print("   📺 实时转录: http://127.0.0.1:5001/realtime.html")
    print("   🎬 文件处理: http://127.0.0.1:5001/app.html")
    print("\n按 Ctrl+C 停止服务")
    
    try:
        # 启动 Flask 应用
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止，谢谢使用!")
    except FileNotFoundError:
        print("\n❌ app.py 文件未找到")
        print("📁 请确保在项目根目录运行此脚本")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("🔧 请检查端口5001是否被占用或查看错误日志")
        sys.exit(1)

if __name__ == "__main__":
    main()