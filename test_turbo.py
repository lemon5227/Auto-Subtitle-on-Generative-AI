#!/usr/bin/env python3
"""
测试 Whisper Large-v3 Turbo 模型
"""

import numpy as np
import torch
from transformers import pipeline

def test_turbo_model():
    """测试 Whisper Large-v3 Turbo 模型"""
    try:
        print("🚀 开始测试 Whisper Large-v3 Turbo...")
        
        # 检查 CUDA 可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"📱 使用设备: {device}")
        print(f"🔢 数据类型: {torch_dtype}")
        
        # 创建 pipeline
        print("📦 正在加载 Whisper Large-v3 Turbo...")
        
        pipe_kwargs = {
            "model": "openai/whisper-large-v3-turbo",
            "torch_dtype": torch_dtype,
            "device": device,
            "chunk_length_s": 30,  # 30秒分块
            "batch_size": 8 if torch.cuda.is_available() else 2,
        }
        
        # 不使用 flash attention，避免依赖问题
        # 如果需要更快的速度，可以手动安装 flash-attn 包
        
        pipe = pipeline("automatic-speech-recognition", **pipe_kwargs)
        print("✅ 模型加载成功!")
        
        # 创建测试音频 (3秒的正弦波，模拟音频)
        print("🎵 创建测试音频...")
        sample_rate = 16000
        duration = 3  # 3 秒
        t = np.linspace(0, duration, int(sample_rate * duration))
        # 创建一个 440Hz 的正弦波 (A4 音符)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # 进行转录测试
        print("🎤 开始转录测试...")
        result = pipe(
            audio,
            chunk_length_s=30,
            batch_size=8 if torch.cuda.is_available() else 2,
            return_timestamps=False
        )
        
        print("✅ 转录完成!")
        print(f"📝 结果: {result}")
        print(f"📄 文本: {result.get('text', 'No text') if isinstance(result, dict) else str(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_turbo_model()
    if success:
        print("\n🎉 Whisper Large-v3 Turbo 测试成功!")
    else:
        print("\n💥 测试失败，请检查错误信息")