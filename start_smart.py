#!/usr/bin/env python3
"""
智能启动器 - AI 字幕生成器
🚀 自动检测并优化 NVIDIA / AMD / Apple Silicon / CPU 环境
"""
import os
import sys
import subprocess
import logging
import socket

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_header():
    """打印启动头部"""
    print("🚀 AI 字幕生成器 - 智能启动器")
    print("=" * 60)
    print("🎯 自动检测最佳计算设备并优化性能配置")
    print("💪 支持: NVIDIA CUDA / AMD ROCm / Apple MPS / CPU")
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

def check_model_availability():
    """检查模型可用性"""
    print("\n📦 检查模型状态:")
    print("-" * 40)
    
    try:
        import whisper
        
        # 检查常用模型
        models_to_check = ['base', 'small', 'medium']
        available_models = []
        
        for model in models_to_check:
            try:
                # 快速检查模型是否已下载
                model_path = whisper._MODELS[model]
                if os.path.exists(os.path.expanduser(f"~/.cache/whisper/{model_path.split('/')[-1]}")):
                    available_models.append(model)
                    print(f"   ✅ {model}")
                else:
                    print(f"   ⏳ {model} (首次使用将自动下载)")
            except:
                print(f"   ❓ {model} (状态未知)")
        
        if available_models:
            print(f"📊 已缓存 {len(available_models)} 个模型")
        else:
            print("💡 首次运行将自动下载所需模型")
            
    except Exception as e:
        logger.warning(f"⚠️ 模型检查失败: {e}")

def get_performance_recommendations(detector):
    """获取性能建议"""
    if not detector:
        return []
        
    recommendations = []
    device_info = detector.device_info
    
    # 基于设备类型给出建议
    if device_info['gpu_type'] == 'nvidia':
        if device_info['memory_gb'] >= 12:
            recommendations.append("💪 显存充足，可使用 large 模型获得最佳效果")
        elif device_info['memory_gb'] >= 8:
            recommendations.append("👍 推荐使用 medium 或 small 模型平衡性能")
        else:
            recommendations.append("⚡ 显存有限，建议使用 base 或 small 模型")
            
    elif device_info['gpu_type'] == 'amd':
        recommendations.append("🎮 AMD GPU 加速，推荐使用 small 或 base 模型")
        recommendations.append("🔧 如遇问题可回退 CPU 模式")
        
    elif device_info['gpu_type'] == 'apple':
        recommendations.append("🍎 Apple Silicon 优化，medium 模型性能良好")
        
    else:
        recommendations.append("⚡ CPU 模式，推荐 base 模型获得最佳速度")
        
    return recommendations

def show_startup_info(detector):
    """显示启动信息"""
    print(f"\n🌐 服务地址:")
    print("-" * 40)
    print(f"   📺 实时转录: http://localhost:5001/realtime.html")
    print(f"   🎬 文件处理: http://localhost:5001/app.html")
    
    # 显示性能建议
    recommendations = get_performance_recommendations(detector)
    if recommendations:
        print(f"\n🎯 性能建议:")
        print("-" * 40)
        for rec in recommendations:
            print(f"   {rec}")
    
    print(f"\n🔔 使用提示:")
    print("-" * 40)
    print(f"   • 首次使用会自动下载模型，请耐心等待")
    print(f"   • 大文件处理可能需要较长时间")
    print(f"   • 按 Ctrl+C 停止服务")
    print(f"   • 遇到问题可查看终端输出信息")

def check_port_availability():
    """检查端口可用性"""
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    if is_port_in_use(5001):
        print("⚠️ 端口 5001 已被占用")
        print("💡 可能已有实例在运行，或其他程序占用了该端口")
        
        response = input("是否继续启动？(y/N): ").lower().strip()
        if response != 'y':
            print("👋 启动已取消")
            return False
    
    return True

def start_application():
    """启动应用"""
    print(f"\n🚀 正在启动 AI 字幕生成器...")
    print("=" * 60)
    
    try:
        # 启动 Flask 应用
        subprocess.run([sys.executable, "app.py"], check=True)
        
    except KeyboardInterrupt:
        print(f"\n\n👋 服务已停止")
        print("感谢使用 AI 字幕生成器！")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 启动失败: 退出码 {e.returncode}")
        show_troubleshooting()
        
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        show_troubleshooting()

def show_troubleshooting():
    """显示故障排除信息"""
    print(f"\n🔧 故障排除:")
    print("-" * 40)
    print(f"   1. 检查依赖: pip install -r requirements.txt")
    print(f"   2. 检查 Python 版本: python --version (需要 3.8+)")
    print(f"   3. 检查端口占用: lsof -i :5001")
    print(f"   4. 查看详细日志: python app.py")
    print(f"   5. 重置环境: 删除虚拟环境重新安装")
    
    print(f"\n🆘 GPU 相关问题:")
    print(f"   • NVIDIA: 检查 CUDA 驱动和 PyTorch CUDA 版本")
    print(f"   • AMD: 确保安装 ROCm 和 PyTorch ROCm 版本") 
    print(f"   • 通用: 可设置环境变量强制 CPU 模式")
    print(f"     export CUDA_VISIBLE_DEVICES=-1")

def main():
    """主函数"""
    print_header()
    
    # 基础检查
    if not check_dependencies():
        return 1
    
    # 端口检查
    if not check_port_availability():
        return 1
        
    # 环境设置和GPU检测
    detector = setup_environment()
    
    # 模型检查
    check_model_availability()
    
    # 显示启动信息
    show_startup_info(detector)
    
    # 启动应用
    start_application()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 启动器异常: {e}")
        sys.exit(1)