#!/usr/bin/env python3
"""
跨平台启动脚本 - Auto Subtitle Generator
支持 Windows, macOS, Linux
"""

import sys
import platform
import subprocess
import shutil
import torch

def check_system():
    """检测系统信息"""
    system = platform.system()
    machine = platform.machine()
    python_version = sys.version_info
    
    print(f"🖥️  系统: {system} {platform.release()}")
    print(f"🔧 架构: {machine}")
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return system, machine

def check_dependencies():
    """检查依赖项"""
    print("\n📦 检查依赖项...")
    
    # 检查 ffmpeg
    if shutil.which('ffmpeg'):
        print("✅ ffmpeg 已安装")
    else:
        print("❌ ffmpeg 未找到")
        system, _ = check_system()
        if system == "Darwin":  # macOS
            print("   安装命令: brew install ffmpeg")
        elif system == "Linux":
            print("   安装命令: sudo apt install ffmpeg  # Ubuntu/Debian")
            print("              sudo yum install ffmpeg  # CentOS/RHEL")
        elif system == "Windows":
            print("   请从 https://ffmpeg.org/ 下载并添加到 PATH")
        return False
    
    # 检查 PyTorch
    try:
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
        print("❌ PyTorch 未安装")
        print("   安装命令:")
        system, machine = check_system()
        if system == "Darwin" and machine == "arm64":  # Apple Silicon
            print("   pip install torch torchvision torchaudio")
        else:
            print("   访问 https://pytorch.org/get-started/locally/ 获取安装命令")
        return False
    
    return True

def main():
    """主函数"""
    print("🎤 Auto Subtitle Generator - 跨平台启动器")
    print("=" * 50)
    
    system, machine = check_system()
    
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装缺少的依赖")
        sys.exit(1)
    
    print("\n🚀 启动应用...")
    
    try:
        # 启动 Flask 应用
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()