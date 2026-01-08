#!/usr/bin/env python3
"""
General GPU Detection and Adaptation Module
Allows the subtitle generator to automatically detect and use the best hardware (CUDA, ROCm, MPS, CPU)
"""
import os
import sys
import subprocess
import logging
from typing import Tuple, Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUDetector:
    """General GPU Detector - Automatically identifies best available compute device"""
    
    def __init__(self):
        self.gpu_info = self._detect_all_gpus()
        self.device_info = self._get_device_info()
        
    def _run_command(self, cmd: str) -> str:
        """Safely execute system command"""
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
        """Detect NVIDIA GPU"""
        nvidia_info = {
            'available': False,
            'gpus': [],
            'driver_version': '',
            'cuda_version': '',
            'total_memory': 0
        }
        
        # Check nvidia-smi
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
        
        # Check CUDA version
        nvcc_output = self._run_command("nvcc --version")
        if nvcc_output:
            for line in nvcc_output.split('\n'):
                if 'release' in line.lower():
                    nvidia_info['cuda_version'] = line.split('release')[-1].strip().split(',')[0]
                    break
        
        return nvidia_info
    
    def _detect_amd_gpu(self) -> Dict:
        """Detect AMD GPU"""
        amd_info = {
            'available': False,
            'gpus': [],
            'rocm_version': '',
            'total_memory': 0
        }
        
        # Detect via lspci
        lspci_output = self._run_command("lspci")
        amd_gpus = []
        for line in lspci_output.split('\n'):
            if 'amd' in line.lower() and ('vga' in line.lower() or 'display' in line.lower()):
                gpu_name = line.split(': ')[-1] if ': ' in line else line
                amd_gpus.append(gpu_name.strip())
        
        if amd_gpus:
            amd_info['available'] = True
            for gpu_name in amd_gpus:
                amd_info['gpus'].append({
                    'name': gpu_name,
                    'memory_mb': 0
                })
        
        # Check ROCm
        if os.path.exists('/opt/rocm'):
            rocm_info = self._run_command("/opt/rocm/bin/rocm_agent_enumerator")
            if rocm_info or os.path.exists('/opt/rocm/bin/rocminfo'):
                amd_info['rocm_version'] = 'Installed'
        
        # Check kernel module
        lsmod_output = self._run_command("lsmod")
        if 'amdgpu' in lsmod_output:
            amd_info['driver_loaded'] = True
        
        return amd_info
    
    def _detect_apple_gpu(self) -> Dict:
        """Detect Apple Silicon GPU"""
        apple_info = {
            'available': False,
            'chip': '',
            'memory': 0
        }
        
        if sys.platform != 'darwin':
            return apple_info
            
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
        return {
            'nvidia': self._detect_nvidia_gpu(),
            'amd': self._detect_amd_gpu(),
            'apple': self._detect_apple_gpu()
        }
    
    def _test_pytorch_device(self, device: str) -> bool:
        """Test PyTorch device availability"""
        try:
            import torch
            
            if device == 'cuda':
                if not torch.cuda.is_available():
                    return False
                try:
                    x = torch.randn(100, 100).cuda()
                    y = torch.matmul(x, x)
                    del x, y
                    torch.cuda.empty_cache()
                    return True
                except Exception as e:
                    logger.warning(f"CUDA test failed: {e}")
                    return False
                    
            elif device == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    return False
                try:
                    x = torch.randn(100, 100).to('mps')
                    y = torch.matmul(x, x)
                    del x, y
                    return True
                except Exception as e:
                    logger.warning(f"MPS test failed: {e}")
                    return False
                    
            elif device == 'cpu':
                return True
                
            return False
            
        except ImportError:
            logger.warning("PyTorch not installed, cannot test device")
            return False
    
    def _get_device_info(self) -> Dict:
        """Get recommended PyTorch device info"""
        device_info = {
            'recommended_device': 'cpu',
            'available_devices': ['cpu'],
            'gpu_type': 'none',
            'gpu_name': '',
            'memory_gb': 0,
            'performance_level': 'basic',
            'optimization_tips': []
        }
        
        # NVIDIA GPU
        if self.gpu_info['nvidia']['available']:
            if self._test_pytorch_device('cuda'):
                device_info['recommended_device'] = 'cuda'
                device_info['available_devices'].append('cuda')
                device_info['gpu_type'] = 'nvidia'
                
                gpus = self.gpu_info['nvidia']['gpus']
                if gpus:
                    best_gpu = max(gpus, key=lambda g: g['memory_mb'])
                    device_info['gpu_name'] = best_gpu['name']
                    device_info['memory_gb'] = best_gpu['memory_mb'] // 1024
                    
                    if device_info['memory_gb'] >= 12:
                        device_info['performance_level'] = 'excellent'
                    elif device_info['memory_gb'] >= 8:
                        device_info['performance_level'] = 'good'
                    elif device_info['memory_gb'] >= 4:
                        device_info['performance_level'] = 'acceptable'
                    else:
                        device_info['performance_level'] = 'limited'
                
                device_info['optimization_tips'] = [
                    "Using CUDA acceleration for best performance",
                    "Large models require sufficient VRAM",
                    "Adjust batch size to optimize VRAM usage"
                ]
        
        # AMD GPU
        elif self.gpu_info['amd']['available']:
            rocm_available = self._test_pytorch_device('cuda') and self.gpu_info['amd'].get('rocm_version')
            
            if rocm_available:
                device_info['recommended_device'] = 'cuda'
                device_info['available_devices'].append('cuda')
                device_info['gpu_type'] = 'amd'
                
                gpus = self.gpu_info['amd']['gpus']
                if gpus:
                    device_info['gpu_name'] = gpus[0]['name']
                    # Estimate based on name
                    gpu_name = gpus[0]['name'].lower()
                    if any(x in gpu_name for x in ['rx 7900', 'rx 7800', 'rx 6900', 'rx 6800']):
                        device_info['performance_level'] = 'excellent'
                        device_info['memory_gb'] = 12
                    elif any(x in gpu_name for x in ['rx 7700', 'rx 7600', 'rx 6700', 'rx 6600']):
                        device_info['performance_level'] = 'good'
                        device_info['memory_gb'] = 8
                    else:
                        device_info['performance_level'] = 'acceptable'
                        device_info['memory_gb'] = 4
                
                device_info['optimization_tips'] = [
                    "Using ROCm for AMD GPU acceleration",
                    "Ensure correct ROCm drivers are installed",
                    "Fallback to CPU if issues arise"
                ]
            else:
                gpus = self.gpu_info['amd']['gpus']
                if gpus:
                    device_info['gpu_name'] = f"{gpus[0]['name']} (ROCm unavailable)"
                
                device_info['optimization_tips'] = [
                    "AMD GPU detected but ROCm unavailable",
                    "Install ROCm drivers/PyTorch ROCm version",
                    "Using CPU mode for stability"
                ]
        
        # Apple Silicon
        elif self.gpu_info['apple']['available']:
            if self._test_pytorch_device('mps'):
                device_info['recommended_device'] = 'mps'
                device_info['available_devices'].append('mps')
                device_info['gpu_type'] = 'apple'
                device_info['gpu_name'] = self.gpu_info['apple']['chip']
                device_info['memory_gb'] = self.gpu_info['apple']['memory']
                device_info['performance_level'] = 'good'
                
                device_info['optimization_tips'] = [
                    "Using Apple Metal Performance Shaders",
                    "Unified memory architecture"
                ]
        
        # CPU
        else:
            device_info['optimization_tips'] = [
                "Using CPU mode",
                "Recommend smaller models for responsiveness",
                "Consider quantized models"
            ]
        
        return device_info
    
    def get_recommended_device(self) -> str:
        return self.device_info['recommended_device']
    
    def get_device_summary(self) -> str:
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
        
        summary += f" - {info['performance_level']} performance"
        
        return summary
    
    def get_optimization_tips(self) -> List[str]:
        return self.device_info['optimization_tips']

def get_optimal_device() -> Tuple[str, Dict]:
    detector = GPUDetector()
    return detector.get_recommended_device(), detector.device_info

def create_device_environment() -> Dict[str, str]:
    detector = GPUDetector()
    device_info = detector.device_info
    env_vars = {}
    
    if device_info['gpu_type'] == 'nvidia':
        env_vars.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        })
    elif device_info['gpu_type'] == 'amd':
        env_vars.update({
            'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
            'ROCM_PATH': '/opt/rocm',
            'HIP_VISIBLE_DEVICES': '0',
            'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:128',
        })
    elif device_info['gpu_type'] == 'apple':
        env_vars.update({
            'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        })
    else:
        env_vars.update({
            'OMP_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4',
        })
    
    env_vars.update({
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTHONUNBUFFERED': '1',
    })
    
    return env_vars
