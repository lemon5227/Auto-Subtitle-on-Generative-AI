#!/usr/bin/env python3
"""
通用 GPU 检测和适配模块
🚀 智能检测并适配 NVIDIA、AMD、Apple Silicon 和 CPU
"""
import os
import sys
import subprocess
import logging
from typing import Tuple, Dict, Optional, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUDetector:
    """通用GPU检测器 - 自动识别最佳可用计算设备"""
    
    def __init__(self):
        self.gpu_info = self._detect_all_gpus()
        self.device_info = self._get_device_info()
        
    def _run_command(self, cmd: str) -> str:
        """安全执行系统命令"""
        try:
            result = subprocess.run(
                cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return ""
    
    def _detect_nvidia_gpu(self) -> Dict:
        """检测 NVIDIA GPU"""
        nvidia_info = {
            'available': False,
            'gpus': [],
            'driver_version': '',
            'cuda_version': '',
            'total_memory': 0
        }
        
        # 检查 nvidia-smi
        nvidia_smi = self._run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
        if nvidia_smi:
            lines = nvidia_smi.split('\n')
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_name = parts[0]
                        memory_mb = int(parts[1]) if parts[1].isdigit() else 0
                        driver_ver = parts[2]
                        
                        nvidia_info['gpus'].append({
                            'name': gpu_name,
                            'memory_mb': memory_mb
                        })
                        nvidia_info['total_memory'] += memory_mb
                        nvidia_info['driver_version'] = driver_ver
                        nvidia_info['available'] = True
        
        # 检查 CUDA 版本
        nvcc_output = self._run_command("nvcc --version")
        if nvcc_output:
            for line in nvcc_output.split('\n'):
                if 'release' in line.lower():
                    nvidia_info['cuda_version'] = line.split('release')[-1].strip().split(',')[0]
                    break
        
        return nvidia_info
    
    def _detect_amd_gpu(self) -> Dict:
        """检测 AMD GPU"""
        amd_info = {
            'available': False,
            'gpus': [],
            'rocm_version': '',
            'total_memory': 0
        }
        
        # 通过 lspci 检测 AMD GPU
        lspci_output = self._run_command("lspci")
        amd_gpus = []
        for line in lspci_output.split('\n'):
            if 'amd' in line.lower() and ('vga' in line.lower() or 'display' in line.lower()):
                # 提取 GPU 名称
                gpu_name = line.split(': ')[-1] if ': ' in line else line
                amd_gpus.append(gpu_name.strip())
        
        if amd_gpus:
            amd_info['available'] = True
            for gpu_name in amd_gpus:
                amd_info['gpus'].append({
                    'name': gpu_name,
                    'memory_mb': 0  # AMD 显存信息较难获取，暂设为0
                })
        
        # 检查 ROCm
        if os.path.exists('/opt/rocm'):
            rocm_info = self._run_command("/opt/rocm/bin/rocm_agent_enumerator")
            if rocm_info or os.path.exists('/opt/rocm/bin/rocminfo'):
                amd_info['rocm_version'] = 'Installed'
        
        # 检查内核模块
        lsmod_output = self._run_command("lsmod")
        if 'amdgpu' in lsmod_output:
            amd_info['driver_loaded'] = True
        
        return amd_info
    
    def _detect_apple_gpu(self) -> Dict:
        """检测 Apple Silicon GPU"""
        apple_info = {
            'available': False,
            'chip': '',
            'memory': 0
        }
        
        # 检查是否为 macOS
        if sys.platform != 'darwin':
            return apple_info
            
        # 检查芯片信息
        system_profiler = self._run_command("system_profiler SPHardwareDataType")
        for line in system_profiler.split('\n'):
            if 'Chip:' in line:
                apple_info['chip'] = line.split('Chip:')[-1].strip()
                if 'M1' in apple_info['chip'] or 'M2' in apple_info['chip'] or 'M3' in apple_info['chip']:
                    apple_info['available'] = True
            elif 'Memory:' in line:
                memory_str = line.split('Memory:')[-1].strip()
                if 'GB' in memory_str:
                    apple_info['memory'] = int(memory_str.split()[0])
        
        return apple_info
    
    def _detect_all_gpus(self) -> Dict:
        """检测所有可用的GPU"""
        return {
            'nvidia': self._detect_nvidia_gpu(),
            'amd': self._detect_amd_gpu(),
            'apple': self._detect_apple_gpu()
        }
    
    def _test_pytorch_device(self, device: str) -> bool:
        """测试 PyTorch 设备可用性"""
        try:
            import torch
            
            if device == 'cuda':
                if not torch.cuda.is_available():
                    return False
                # 简单的 CUDA 测试
                try:
                    x = torch.randn(100, 100).cuda()
                    y = torch.matmul(x, x)
                    del x, y
                    torch.cuda.empty_cache()
                    return True
                except Exception as e:
                    logger.warning(f"CUDA测试失败: {e}")
                    return False
                    
            elif device == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    return False
                # 简单的 MPS 测试
                try:
                    x = torch.randn(100, 100).to('mps')
                    y = torch.matmul(x, x)
                    del x, y
                    return True
                except Exception as e:
                    logger.warning(f"MPS测试失败: {e}")
                    return False
                    
            elif device == 'cpu':
                return True
                
            return False
            
        except ImportError:
            logger.warning("PyTorch 未安装，无法测试设备")
            return False
    
    def _get_device_info(self) -> Dict:
        """获取推荐的 PyTorch 设备信息"""
        device_info = {
            'recommended_device': 'cpu',
            'available_devices': ['cpu'],
            'gpu_type': 'none',
            'gpu_name': '',
            'memory_gb': 0,
            'performance_level': 'basic',
            'optimization_tips': []
        }
        
        # NVIDIA GPU (最高优先级)
        if self.gpu_info['nvidia']['available']:
            if self._test_pytorch_device('cuda'):
                device_info['recommended_device'] = 'cuda'
                device_info['available_devices'].append('cuda')
                device_info['gpu_type'] = 'nvidia'
                
                # 获取最佳 GPU 信息
                gpus = self.gpu_info['nvidia']['gpus']
                if gpus:
                    best_gpu = max(gpus, key=lambda g: g['memory_mb'])
                    device_info['gpu_name'] = best_gpu['name']
                    device_info['memory_gb'] = best_gpu['memory_mb'] // 1024
                    
                    # 性能等级判断
                    if device_info['memory_gb'] >= 12:
                        device_info['performance_level'] = 'excellent'
                    elif device_info['memory_gb'] >= 8:
                        device_info['performance_level'] = 'good'
                    elif device_info['memory_gb'] >= 4:
                        device_info['performance_level'] = 'acceptable'
                    else:
                        device_info['performance_level'] = 'limited'
                
                device_info['optimization_tips'] = [
                    "使用 CUDA 加速获得最佳性能",
                    "大模型需要足够显存，建议选择合适的模型大小",
                    "可调整批处理大小优化显存使用"
                ]
        
        # AMD GPU (第二优先级，ROCm 支持)
        elif self.gpu_info['amd']['available']:
            # 测试ROCm支持
            rocm_available = self._test_pytorch_device('cuda') and self.gpu_info['amd'].get('rocm_version')
            
            if rocm_available:
                device_info['recommended_device'] = 'cuda'  # ROCm 通过 CUDA API
                device_info['available_devices'].append('cuda')
                device_info['gpu_type'] = 'amd'
                
                gpus = self.gpu_info['amd']['gpus']
                if gpus:
                    device_info['gpu_name'] = gpus[0]['name']
                    
                    # AMD GPU 性能预估 (基于型号)
                    gpu_name = gpus[0]['name'].lower()
                    if any(x in gpu_name for x in ['rx 7900', 'rx 7800', 'rx 6900', 'rx 6800']):
                        device_info['performance_level'] = 'excellent'
                        device_info['memory_gb'] = 12  # 估算
                    elif any(x in gpu_name for x in ['rx 7700', 'rx 7600', 'rx 6700', 'rx 6600']):
                        device_info['performance_level'] = 'good'
                        device_info['memory_gb'] = 8   # 估算
                    elif any(x in gpu_name for x in ['rx 5700', 'rx 5600', 'rx 5500']):
                        device_info['performance_level'] = 'acceptable'
                        device_info['memory_gb'] = 6   # 估算
                    else:
                        device_info['performance_level'] = 'limited'
                        device_info['memory_gb'] = 4   # 估算
                
                device_info['optimization_tips'] = [
                    "使用 ROCm 获得 AMD GPU 加速",
                    "确保安装了正确的 ROCm 驱动",
                    "部分功能可能需要特定的环境变量",
                    "如遇问题可回退到 CPU 模式"
                ]
            else:
                # AMD GPU存在但ROCm不可用，记录信息但回退到CPU
                gpus = self.gpu_info['amd']['gpus']
                if gpus:
                    device_info['gpu_name'] = f"{gpus[0]['name']} (ROCm不可用)"
                
                device_info['optimization_tips'] = [
                    "检测到AMD GPU但ROCm不可用",
                    "安装ROCm驱动: sudo apt install rocm-dev",
                    "安装PyTorch ROCm版本获得GPU加速",
                    "当前使用CPU模式确保稳定运行"
                ]
        
        # Apple Silicon GPU (第三优先级)
        elif self.gpu_info['apple']['available']:
            if self._test_pytorch_device('mps'):
                device_info['recommended_device'] = 'mps'
                device_info['available_devices'].append('mps')
                device_info['gpu_type'] = 'apple'
                device_info['gpu_name'] = self.gpu_info['apple']['chip']
                device_info['memory_gb'] = self.gpu_info['apple']['memory']
                device_info['performance_level'] = 'good'
                
                device_info['optimization_tips'] = [
                    "使用 Apple Metal Performance Shaders 加速",
                    "在 Apple Silicon Mac 上获得良好性能",
                    "统一内存架构，显存即系统内存"
                ]
        
        # CPU 模式 (最后选择)
        else:
            device_info['optimization_tips'] = [
                "当前使用 CPU 模式",
                "推荐选择较小的模型以获得更好响应速度",
                "考虑使用量化模型减少计算量",
                "多线程设置: export OMP_NUM_THREADS=4"
            ]
        
        return device_info
    
    def get_recommended_device(self) -> str:
        """获取推荐的计算设备"""
        return self.device_info['recommended_device']
    
    def get_device_summary(self) -> str:
        """获取设备信息摘要"""
        info = self.device_info
        gpu_type_map = {
            'nvidia': '🟢 NVIDIA GPU',
            'amd': '🔴 AMD GPU', 
            'apple': '🟡 Apple Silicon',
            'none': '🔵 CPU'
        }
        
        summary = f"{gpu_type_map.get(info['gpu_type'], '🔵 CPU')}"
        
        if info['gpu_name']:
            summary += f" ({info['gpu_name']}"
            if info['memory_gb'] > 0:
                summary += f", {info['memory_gb']}GB"
            summary += ")"
        
        summary += f" - {info['performance_level']} 性能"
        
        return summary
    
    def get_optimization_tips(self) -> List[str]:
        """获取优化建议"""
        return self.device_info['optimization_tips']
    
    def print_detection_report(self):
        """打印完整的检测报告"""
        print("🚀 GPU 检测和设备选择报告")
        print("=" * 50)
        
        # NVIDIA 信息
        nvidia = self.gpu_info['nvidia']
        print(f"🟢 NVIDIA GPU: {'✅ 检测到' if nvidia['available'] else '❌ 未检测到'}")
        if nvidia['available']:
            for i, gpu in enumerate(nvidia['gpus']):
                print(f"   GPU {i+1}: {gpu['name']} ({gpu['memory_mb']}MB)")
            if nvidia['driver_version']:
                print(f"   驱动版本: {nvidia['driver_version']}")
            if nvidia['cuda_version']:
                print(f"   CUDA 版本: {nvidia['cuda_version']}")
        
        # AMD 信息  
        amd = self.gpu_info['amd']
        print(f"🔴 AMD GPU: {'✅ 检测到' if amd['available'] else '❌ 未检测到'}")
        if amd['available']:
            for i, gpu in enumerate(amd['gpus']):
                print(f"   GPU {i+1}: {gpu['name']}")
            if amd.get('rocm_version'):
                print(f"   ROCm: {amd['rocm_version']}")
        
        # Apple 信息
        apple = self.gpu_info['apple']
        print(f"🟡 Apple GPU: {'✅ 检测到' if apple['available'] else '❌ 未检测到'}")
        if apple['available']:
            print(f"   芯片: {apple['chip']}")
            if apple['memory'] > 0:
                print(f"   内存: {apple['memory']}GB")
        
        print(f"\n🎯 推荐设备: {self.get_recommended_device()}")
        print(f"📊 设备摘要: {self.get_device_summary()}")
        
        print(f"\n💡 优化建议:")
        for tip in self.get_optimization_tips():
            print(f"   • {tip}")


def get_optimal_device() -> Tuple[str, Dict]:
    """
    获取最佳计算设备
    返回: (device_name, device_info)
    """
    detector = GPUDetector()
    return detector.get_recommended_device(), detector.device_info


def create_device_environment() -> Dict[str, str]:
    """创建针对检测到的设备的环境变量"""
    detector = GPUDetector()
    device_info = detector.device_info
    env_vars = {}
    
    if device_info['gpu_type'] == 'nvidia':
        # NVIDIA CUDA 优化
        env_vars.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        })
    elif device_info['gpu_type'] == 'amd':
        # AMD ROCm 优化
        env_vars.update({
            'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
            'ROCM_PATH': '/opt/rocm',
            'HIP_VISIBLE_DEVICES': '0',
            'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:128',
        })
    elif device_info['gpu_type'] == 'apple':
        # Apple MPS 优化
        env_vars.update({
            'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        })
    else:
        # CPU 优化
        env_vars.update({
            'OMP_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4',
        })
    
    # 通用优化
    env_vars.update({
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTHONUNBUFFERED': '1',
    })
    
    return env_vars


if __name__ == "__main__":
    # 测试脚本
    detector = GPUDetector()
    detector.print_detection_report()
    
    print(f"\n🔧 推荐环境变量:")
    env_vars = create_device_environment()
    for key, value in env_vars.items():
        print(f"   export {key}={value}")