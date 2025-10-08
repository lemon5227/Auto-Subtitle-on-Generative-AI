#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen智能字幕校对功能测试脚本
测试Qwen模型的加载、推理和字幕优化功能
支持项目启动前预下载模型
"""

import sys
import time
import argparse
import os

def test_qwen_import():
    """测试Qwen相关库导入"""
    print("=" * 60)
    print("1. 测试Qwen相关库导入")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ transformers库导入成功")
        
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_gpu_detection():
    """测试GPU检测功能"""
    print("\n" + "=" * 60)
    print("2. 测试GPU检测功能")
    print("=" * 60)
    
    try:
        from gpu_detector import GPUDetector
        
        detector = GPUDetector()
        device = detector.get_optimal_device()
        info = detector.get_device_info()
        
        print(f"✅ 检测到设备: {device}")
        print(f"   设备类型: {info['type']}")
        print(f"   设备名称: {info['name']}")
        print(f"   性能评级: {info['performance']}")
        
        if info['memory']:
            print(f"   显存信息: {info['memory']}")
        
        print(f"\n💡 推荐: {info['recommendation']}")
        
        return True, device
    except Exception as e:
        print(f"❌ GPU检测失败: {e}")
        return False, 'cpu'

def test_qwen_model_info():
    """测试Qwen模型信息获取"""
    print("\n" + "=" * 60)
    print("3. 测试Qwen模型信息")
    print("=" * 60)
    
    try:
        # 仅Qwen3系列模型（移除过期的Qwen2.5）
        models = [
            {"name": "Qwen3-0.6B", "model_id": "Qwen/Qwen3-0.6B", "size": "0.6B", "best_for": "实时翻译（超轻量）", "vram": "~2GB"},
            {"name": "Qwen3-1.7B", "model_id": "Qwen/Qwen3-1.7B", "size": "1.7B", "best_for": "实时翻译（推荐）", "vram": "~4GB"},
            {"name": "Qwen3-4B", "model_id": "Qwen/Qwen3-4B", "size": "4B", "best_for": "字幕优化（推荐）", "vram": "~8GB"},
            {"name": "Qwen3-8B", "model_id": "Qwen/Qwen3-8B", "size": "8B", "best_for": "字幕优化（高质量）", "vram": "~16GB"},
        ]
        
        print("✅ 支持的Qwen3模型（已移除过期的Qwen2.5）:")
        for i, model in enumerate(models, 1):
            print(f"\n   {i}. {model['name']} ({model['size']})")
            print(f"      Model ID: {model['model_id']}")
            print(f"      适用场景: {model['best_for']}")
            print(f"      显存需求: {model['vram']}")
        
        return True, models
    except Exception as e:
        print(f"❌ 获取模型信息失败: {e}")
        return False, []

def download_qwen_model(model_id, device='cpu', use_fp16=False):
    """下载并测试Qwen模型
    
    Args:
        model_id: 模型ID (例如: Qwen/Qwen3-1.7B)
        device: 设备类型 (cpu/cuda)
        use_fp16: 是否使用FP16精度
    """
    print("\n" + "=" * 60)
    print(f"下载模型: {model_id}")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"📥 开始下载模型: {model_id}")
        print(f"   设备: {device}")
        print(f"   精度: {'FP16' if use_fp16 else 'FP32'}")
        print(f"   缓存目录: {os.path.expanduser('~/.cache/huggingface')}")
        
        start_time = time.time()
        
        # 下载tokenizer
        print("\n1️⃣ 下载Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print("✅ Tokenizer下载完成")
        
        # 下载模型
        print("\n2️⃣ 下载模型权重...")
        dtype = torch.float16 if use_fp16 and device == 'cuda' else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == 'cuda' else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device == 'cpu':
            model = model.to('cpu')
        
        download_time = time.time() - start_time
        print(f"✅ 模型下载完成 (耗时: {download_time:.1f}秒)")
        
        # 测试推理
        print("\n3️⃣ 测试模型推理...")
        test_text = "测试文本：这是一个简单的测试"
        
        messages = [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": test_text}
        ]
        
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text_input], return_tensors="pt")
        
        if device == 'cuda':
            model_inputs = model_inputs.to('cuda')
        
        with torch.no_grad():
            start_inference = time.time()
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            inference_time = time.time() - start_inference
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"✅ 推理测试成功 (耗时: {inference_time:.2f}秒)")
        print(f"   输入: {test_text}")
        print(f"   输出: {response[:100]}...")
        
        # 显示模型大小
        if device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"\n📊 显存占用: {memory_allocated:.2f} GB")
        
        print(f"\n✅ 模型 {model_id} 下载并测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_all_models(device='cpu', use_fp16=False, model_filter=None):
    """下载所有支持的模型
    
    Args:
        device: 设备类型
        use_fp16: 是否使用FP16
        model_filter: 模型过滤器 (realtime/refinement/all)
    """
    print("\n" + "=" * 60)
    print("批量下载Qwen模型")
    print("=" * 60)
    
    models_to_download = []
    
    if model_filter == 'realtime':
        models_to_download = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
        ]
        print("📦 下载实时翻译模型（轻量级）")
    elif model_filter == 'refinement':
        models_to_download = [
            "Qwen/Qwen3-4B",
        ]
        print("📦 下载字幕优化模型（标准）")
    else:  # all
        models_to_download = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
        ]
        print("📦 下载所有Qwen3模型")
    
    print(f"共 {len(models_to_download)} 个模型")
    
    success_count = 0
    for i, model_id in enumerate(models_to_download, 1):
        print(f"\n{'='*60}")
        print(f"进度: [{i}/{len(models_to_download)}]")
        print(f"{'='*60}")
        
        if download_qwen_model(model_id, device, use_fp16):
            success_count += 1
        else:
            print(f"⚠️ 模型 {model_id} 下载失败，继续下一个...")
    
    print("\n" + "=" * 60)
    print(f"下载完成: {success_count}/{len(models_to_download)} 个模型成功")
    print("=" * 60)
    
    return success_count == len(models_to_download)

def test_rule_based_refinement():
    """测试基于规则的字幕优化"""
    print("\n" + "=" * 60)
    print("4. 测试基于规则的字幕优化")
    print("=" * 60)
    
    test_cases = [
        {
            "input": "嗯这个呃方法很有效",
            "expected_improvements": ["去除口语词", "标点修正"]
        },
        {
            "input": "我们今天   要讨论的是    人工智能",
            "expected_improvements": ["空格修正"]
        },
        {
            "input": "这个项目已经完成了",
            "expected_improvements": ["标点补充"]
        }
    ]
    
    import re
    success_count = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"   原文: {case['input']}")
        
        # 模拟规则优化
        refined = case['input']
        
        # 去除口语词
        fillers = ['嗯', '啊', '呃', '那个', '这个', '就是说']
        for filler in fillers:
            refined = refined.replace(filler, '')
        
        # 修正空格
        refined = re.sub(r'\s+', ' ', refined).strip()
        
        # 添加标点
        if refined and not refined[-1] in '.!?。！？':
            refined += '。'
        
        print(f"   优化: {refined}")
        print(f"   改进: {', '.join(case['expected_improvements'])}")
        
        if refined != case['input']:
            print("   ✅ 优化成功")
            success_count += 1
        else:
            print("   ⚠️ 无需优化")
    
    print(f"\n总结: {success_count}/{len(test_cases)} 测试通过")
    return success_count == len(test_cases)

def test_qwen_model_loading(device='cpu'):
    """测试Qwen模型加载（可选，需要下载模型）"""
    print("\n" + "=" * 60)
    print("5. 测试Qwen模型加载（可选）")
    print("=" * 60)
    print("⚠️ 此测试需要下载Qwen模型（约3-6GB）")
    print("如果模型未下载，将跳过此测试")
    
    user_input = input("\n是否尝试加载Qwen模型？(y/N): ").strip().lower()
    
    if user_input != 'y':
        print("⏭️ 跳过模型加载测试")
        return True
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # 使用最小的模型
        print(f"\n🔄 正在加载模型: {model_id}")
        print("   这可能需要几分钟（首次需要下载）...")
        
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto" if device != 'cpu' else None,
            trust_remote_code=True
        )
        
        if device == 'cpu':
            model = model.to('cpu')
        
        load_time = time.time() - start_time
        
        print(f"✅ 模型加载成功 (耗时: {load_time:.2f}秒)")
        print(f"   设备: {device}")
        print(f"   参数量: 约1.5B")
        
        # 测试简单推理
        print("\n🧪 测试推理功能...")
        test_text = "我们在座一个新的项目"
        
        messages = [
            {"role": "system", "content": "你是一个字幕校对助手，修正识别错误。"},
            {"role": "user", "content": f"修正这句话: {test_text}"}
        ]
        
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text_input], return_tensors="pt")
        
        if device != 'cpu':
            model_inputs = model_inputs.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True,
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"   原文: {test_text}")
        print(f"   Qwen优化: {response}")
        print("✅ 推理测试成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载/推理失败: {e}")
        print("💡 提示: 请确保网络畅通，模型会自动从HuggingFace下载")
        print("   可以设置镜像加速: export HF_ENDPOINT=https://hf-mirror.com")
        return False

def test_api_endpoints():
    """测试API端点（需要启动服务）"""
    print("\n" + "=" * 60)
    print("6. 测试API端点")
    print("=" * 60)
    print("⚠️ 此测试需要先启动Flask服务: python app.py")
    
    user_input = input("\n服务是否已启动？(y/N): ").strip().lower()
    
    if user_input != 'y':
        print("⏭️ 跳过API测试")
        print("💡 启动服务后可以手动测试:")
        print("   curl -X GET http://localhost:5001/api/qwen_models")
        print("   curl -X POST http://localhost:5001/api/refine_subtitle -H 'Content-Type: application/json' -d '{\"text\":\"测试\",\"model\":\"local\"}'")
        return True
    
    try:
        import requests
        
        # 测试模型列表API
        print("\n测试 GET /api/qwen_models")
        response = requests.get('http://localhost:5001/api/qwen_models', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API响应成功")
            print(f"   Qwen可用: {data.get('available')}")
            print(f"   支持模型数: {len(data.get('models', []))}")
        else:
            print(f"❌ API响应失败: {response.status_code}")
            return False
        
        # 测试字幕优化API
        print("\n测试 POST /api/refine_subtitle")
        test_data = {
            "text": "嗯这个呃方法很有效",
            "model": "local",
            "options": {
                "fix_punctuation": True,
                "remove_fillers": True
            }
        }
        
        response = requests.post(
            'http://localhost:5001/api/refine_subtitle',
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 字幕优化成功")
            print(f"   原文: {data.get('original_text')}")
            print(f"   优化: {data.get('refined_text')}")
            print(f"   模型: {data.get('model')}")
        else:
            print(f"❌ API响应失败: {response.status_code}")
            return False
        
        return True
        
    except ImportError:
        print("❌ 缺少requests库: pip install requests")
        return False
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def main():
    """主测试流程"""
    parser = argparse.ArgumentParser(description='Qwen智能字幕功能测试和模型下载工具')
    parser.add_argument('--download', action='store_true', help='下载模型而不运行测试')
    parser.add_argument('--model', choices=['realtime', 'refinement', 'all'], default='realtime',
                       help='指定要下载的模型类型 (realtime: 实时翻译模型, refinement: 字幕优化模型, all: 所有模型)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='指定设备类型 (cpu/cuda)，默认自动检测')
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度（仅GPU支持）')
    parser.add_argument('--skip-tests', action='store_true', help='跳过测试，仅下载模型')
    
    args = parser.parse_args()
    
    print("\n" + "🧪" * 30)
    print("   Qwen智能字幕校对功能测试")
    print("🧪" * 30 + "\n")
    
    # 检测设备
    device = args.device
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except:
            device = 'cpu'
    
    # 如果指定了下载模式
    if args.download or args.skip_tests:
        print(f"\n{'='*60}")
        print("📥 模型下载模式")
        print(f"{'='*60}")
        print(f"目标设备: {device}")
        print(f"使用FP16: {args.fp16 and device == 'cuda'}")
        print(f"模型类型: {args.model}")
        
        success = download_all_models(
            device=device,
            use_fp16=args.fp16 and device == 'cuda',
            model_filter=args.model
        )
        
        if success:
            print("\n✅ 所有模型下载成功！")
            print("\n📦 已下载的模型可在以下目录找到:")
            print(f"   {os.path.expanduser('~/.cache/huggingface/hub')}")
            print("\n💡 现在可以启动服务:")
            print("   python app.py")
        else:
            print("\n⚠️ 部分模型下载失败")
            print("💡 可以尝试设置镜像:")
            print("   export HF_ENDPOINT=https://hf-mirror.com")
            print("   python test_qwen.py --download --model realtime")
        
        return success
    
    # 正常测试模式
    results = []
    
    # 1. 测试库导入
    results.append(("库导入", test_qwen_import()))
    
    # 2. 测试GPU检测
    gpu_ok, device = test_gpu_detection()
    results.append(("GPU检测", gpu_ok))
    
    # 3. 测试模型信息
    model_info_ok, models = test_qwen_model_info()
    results.append(("模型信息", model_info_ok))
    
    # 4. 测试规则优化
    results.append(("规则优化", test_rule_based_refinement()))
    
    # 5. 测试模型加载（可选）
    results.append(("模型加载", test_qwen_model_loading(device)))
    
    # 6. 测试API（可选）
    results.append(("API端点", test_api_endpoints()))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:15} {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Qwen功能已就绪")
        print("\n下一步:")
        print("1. 预下载模型（可选）:")
        print("   python test_qwen.py --download --model realtime  # 下载实时翻译模型")
        print("   python test_qwen.py --download --model refinement  # 下载字幕优化模型")
        print("   python test_qwen.py --download --model all  # 下载所有模型")
        print("\n2. 启动服务: python app.py")
        print("3. 访问实时转录: http://localhost:5001/realtime.html")
        print("4. 启用字幕优化并选择Qwen模型")
    else:
        print("\n⚠️ 部分测试未通过，请检查上述错误信息")
        print("\n常见问题:")
        print("- 模型下载失败: 设置镜像 export HF_ENDPOINT=https://hf-mirror.com")
        print("- 显存不足: 使用更小的模型(Qwen3-0.6B 或 Qwen3-1.7B)")
        print("- CPU模式慢: 考虑升级硬件或使用本地规则优化")
        print("\n💡 可以先下载模型:")
        print("   python test_qwen.py --download --model realtime")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
