#!/usr/bin/env python3
"""
GPU检测测试脚本
🧪 测试通用GPU检测和适配系统
"""
import sys
import os

def test_gpu_detector():
    """测试GPU检测器功能"""
    print("🧪 GPU 检测器测试")
    print("=" * 50)
    
    try:
        from gpu_detector import GPUDetector, get_optimal_device, create_device_environment
        
        # 创建检测器
        detector = GPUDetector()
        
        # 打印完整检测报告
        detector.print_detection_report()
        
        # 测试最佳设备选择
        print(f"\n🎯 设备选择测试:")
        print("-" * 30)
        device, device_info = get_optimal_device()
        print(f"推荐设备: {device}")
        print(f"设备类型: {device_info['gpu_type']}")
        print(f"性能等级: {device_info['performance_level']}")
        
        # 测试环境变量
        print(f"\n🔧 环境变量测试:")
        print("-" * 30)
        env_vars = create_device_environment()
        for key, value in list(env_vars.items())[:5]:  # 只显示前5个
            print(f"   {key}={value}")
        if len(env_vars) > 5:
            print(f"   ... 共 {len(env_vars)} 个环境变量")
        
        print(f"\n✅ GPU检测器测试完成")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_pytorch_integration():
    """测试PyTorch集成"""
    print(f"\n🔥 PyTorch 集成测试")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        
        # 测试设备可用性
        devices_tested = []
        
        # CUDA测试
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            devices_tested.append('cuda')
            
            # 简单计算测试
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.matmul(x, x)
                print(f"   CUDA 计算测试: ✅ 通过")
                del x, y
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   CUDA 计算测试: ❌ 失败 - {e}")
        else:
            print(f"❌ CUDA 不可用")
        
        # MPS测试 (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"✅ MPS (Apple Silicon) 可用")
            devices_tested.append('mps')
            
            try:
                x = torch.randn(1000, 1000).to('mps')
                y = torch.matmul(x, x)
                print(f"   MPS 计算测试: ✅ 通过")
                del x, y
            except Exception as e:
                print(f"   MPS 计算测试: ❌ 失败 - {e}")
        else:
            print(f"❌ MPS 不可用")
        
        # CPU测试
        print(f"✅ CPU 可用")
        devices_tested.append('cpu')
        try:
            x = torch.randn(1000, 1000)
            y = torch.matmul(x, x)
            print(f"   CPU 计算测试: ✅ 通过")
            del x, y
        except Exception as e:
            print(f"   CPU 计算测试: ❌ 失败 - {e}")
        
        print(f"\n📊 可用设备: {', '.join(devices_tested)}")
        return True
        
    except ImportError:
        print(f"❌ PyTorch 未安装")
        return False
    except Exception as e:
        print(f"❌ PyTorch 测试失败: {e}")
        return False

def test_whisper_compatibility():
    """测试Whisper兼容性"""
    print(f"\n🎤 Whisper 兼容性测试")
    print("=" * 50)
    
    try:
        import whisper
        print(f"✅ Whisper 已安装")
        
        # 获取可用模型
        models = list(whisper._MODELS.keys())
        print(f"📦 可用模型: {', '.join(models)}")
        
        # 检查缓存的模型
        cache_dir = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(cache_dir):
            cached_files = os.listdir(cache_dir)
            cached_models = [f for f in cached_files if f.endswith('.pt')]
            if cached_models:
                print(f"💾 已缓存模型: {len(cached_models)} 个")
            else:
                print(f"💾 暂无缓存模型")
        else:
            print(f"💾 缓存目录不存在")
        
        return True
        
    except ImportError:
        print(f"❌ Whisper 未安装")
        return False
    except Exception as e:
        print(f"❌ Whisper 测试失败: {e}")
        return False

def test_environment_setup():
    """测试环境设置"""
    print(f"\n🌍 环境设置测试")
    print("=" * 50)
    
    # 检查当前环境变量
    gpu_related_vars = [
        'CUDA_VISIBLE_DEVICES', 'ROCM_PATH', 'HSA_OVERRIDE_GFX_VERSION',
        'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS'
    ]
    
    print("🔍 当前GPU相关环境变量:")
    for var in gpu_related_vars:
        value = os.environ.get(var)
        if value:
            print(f"   {var}={value}")
        else:
            print(f"   {var}=未设置")
    
    return True

def main():
    """主测试函数"""
    print("🧪 AI 字幕生成器 - GPU 检测系统测试")
    print("=" * 60)
    
    tests = [
        ("GPU检测器", test_gpu_detector),
        ("PyTorch集成", test_pytorch_integration), 
        ("Whisper兼容性", test_whisper_compatibility),
        ("环境设置", test_environment_setup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果摘要
    print(f"\n🏁 测试结果摘要")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 通过率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print(f"🎉 所有测试通过！系统准备就绪")
        return 0
    else:
        print(f"⚠️ 部分测试失败，请检查环境配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())