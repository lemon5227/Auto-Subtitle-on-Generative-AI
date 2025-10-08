#!/usr/bin/env python3
"""测试优化后的翻译速度"""

import sys
import time
import torch

# 设置环境
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("🔧 导入模块...")
from transformers import AutoModelForCausalLM, AutoTokenizer

print("✅ 模块导入完成\n")

# 测试参数
MODEL_ID = "Qwen/Qwen3-0.6B"
TEST_TEXTS = [
    "Hello, how are you?",
    "I don't know how to do it.",
    "Yeah, I agree.",
]

print(f"📦 加载模型: {MODEL_ID}")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

load_time = time.time() - start_time
print(f"✅ 模型加载完成，耗时: {load_time:.2f}秒\n")

# 新的简洁prompt
SYSTEM_PROMPT = """你是专业字幕翻译助手。直接输出简洁的中文翻译。

规则：
- 只输出中文翻译
- 不要解释、不要思考过程
- 保持自然流畅
- 保留语气和情感"""

# 测试翻译
print("=" * 60)
print("开始翻译测试（优化prompt）")
print("=" * 60)

for i, text in enumerate(TEST_TEXTS, 1):
    print(f"\n【测试 {i}/{len(TEST_TEXTS)}】")
    print(f"原文: {text}")
    
    # 构建prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"翻译成中文：\n{text}"}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    # 生成翻译
    start = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    elapsed = time.time() - start
    
    # 解码
    output_ids = [
        output_id[len(input_id):] 
        for input_id, output_id in zip(model_inputs.input_ids, generated_ids)
    ]
    
    translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # 清理 <think> 标签
    if '<think>' in translation:
        parts = translation.split('</think>')
        if len(parts) > 1:
            translation = parts[1].strip()
    
    print(f"译文: {translation}")
    print(f"⏱️  耗时: {elapsed:.3f}秒")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
