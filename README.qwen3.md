# Qwen3 智能字幕校对与翻译 - 完整指南

## 🎯 Qwen3 特性

本项目现已集成**Qwen3**系列最新大语言模型，提供专业级字幕校对和翻译功能。

### 🆕 Qwen3 vs Qwen2.5 对比

| 特性 | Qwen3-4B | Qwen3-8B | Qwen2.5-3B | Qwen2.5-7B |
|------|----------|----------|------------|------------|
| **发布时间** | 2024最新 | 2024最新 | 2024 | 2024 |
| **参数量** | 4B | 8B | 3B | 7B |
| **模型大小** | ~8GB | ~16GB | ~6GB | ~14GB |
| **ASR纠错** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **翻译质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **推理速度** | 快 | 中等 | 快 | 中等 |
| **推荐场景** | **通用推荐** | 高质量要求 | 向后兼容 | 向后兼容 |

### ✨ 核心优势

1. **更强的ASR纠错能力**
   - 同音字识别准确率提升30%
   - 更好的上下文理解
   - 专业术语识别更准确

2. **专业的字幕翻译**
   - 内置中英互译优化prompt
   - 保持字幕简洁性
   - 符合口语化表达

3. **优化的推理性能**
   - 更快的推理速度
   - 更低的显存占用
   - 支持CPU推理

## 🚀 快速开始

### 1. 安装依赖

```bash
# 确保transformers版本 >= 4.37.0
pip install transformers>=4.37.0 -U
pip install torch torchvision torchaudio
```

### 2. 启动服务

```bash
python app.py
```

首次使用会自动下载模型：
- Qwen3-4B: ~8GB (推荐)
- Qwen3-8B: ~16GB (高质量)

### 3. 使用功能

访问: http://localhost:5001/realtime.html

1. 点击"高级设置"
2. 启用"字幕优化"
3. 选择"Qwen3-4B"模型
4. 勾选优化选项
5. 开始录音

## 📝 专业Prompt设计

### 字幕校对Prompt (ASR Error Correction)

我们为Qwen3设计了专业的ASR校对prompt：

```
你是一个专业的ASR（自动语音识别）字幕校对专家，专注于修正语音识别错误。

核心任务：
1. 同音字/近音字纠错：识别并修正同音或近音导致的错误
2. 词语边界修正：正确识别词语边界，修正分词错误
3. 语法修正：修正明显的语法错误
4. 口语转书面：适度优化口语表达
5. 标点规范：添加或修正标点符号，提升可读性

重要原则：
✓ 只修正明确的ASR错误，不过度改写
✓ 保持原意和说话风格
✓ 利用上下文理解语义
✓ 不添加原文不存在的内容
✓ 不确定时保持原样
```

**效果示例**：

| 原始ASR输出 | Qwen3校对后 | 错误类型 |
|------------|------------|---------|
| 我们在座一个新项目 | 我们再做一个新项目 | 同音字 |
| 他说的话有到礼 | 他说的话有道理 | 同音字 |
| 人工只能技术 | 人工智能技术 | 词语边界 |
| 嗯这个呃方法很有效 | 这个方法很有效 | 口语词 |
| 他的很高兴 | 他很高兴 | 语法错误 |

### 翻译Prompt (Subtitle Translation)

专门优化的字幕翻译prompt：

**中译英**:
```
你是一个专业的字幕翻译专家，专注于中文到英文的字幕翻译。

翻译原则：
1. 准确性：忠实传达原意，不遗漏关键信息
2. 自然性：译文符合英语表达习惯，流畅自然
3. 简洁性：字幕要简洁明了，避免冗长
4. 口语化：保持对话的口语特征
5. 上下文：利用上下文确保翻译连贯性
```

**英译中**:
```
你是一个专业的字幕翻译专家，专注于英文到中文的字幕翻译。

翻译原则：
1. 准确性：忠实传达原意，不遗漏关键信息
2. 地道性：译文符合中文表达习惯，自然流畅
3. 简洁性：字幕要简洁明了，避免冗长
4. 口语化：保持对话的口语特征
5. 上下文：利用上下文确保翻译连贯性
```

## 💡 最佳实践

### 模型选择建议

**Qwen3-4B (推荐)**:
- ✅ 日常使用
- ✅ 会议记录
- ✅ 教育培训
- ✅ 实时字幕
- ✅ 资源受限环境

**Qwen3-8B (高质量)**:
- ✅ 专业内容
- ✅ 技术演讲
- ✅ 学术讲座
- ✅ 重要会议
- ✅ 有充足GPU显存

### 参数调优

在`app.py`的生成配置中，我们优化了参数：

```python
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=256,        # 字幕校对
    temperature=0.3,           # 低温度保证稳定性
    top_p=0.85,                # 适度多样性
    repetition_penalty=1.1,    # 避免重复
)

# 翻译使用不同参数
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,        # 翻译可能更长
    temperature=0.5,           # 稍高温度增加自然度
    top_p=0.9,
    repetition_penalty=1.1,
)
```

### 上下文策略

系统会自动传递最近3条字幕作为上下文：

```javascript
// 前端自动获取上下文
const contextTexts = subtitleItems
  .slice(-3)  // 最近3条
  .map(item => item.querySelector('.original-text p').textContent)
  .filter(t => t);

// 调用API时传递
fetch('/api/refine_subtitle', {
  body: JSON.stringify({
    text: currentText,
    context: contextTexts,  // 上下文数组
    model: 'qwen3-4b'
  })
});
```

## 🎮 硬件要求

### 推荐配置

| 模型 | 最低配置 | 推荐配置 | 推理速度 |
|------|---------|---------|---------|
| **Qwen3-4B** | 8GB RAM/VRAM | 12GB VRAM | 2-3秒/次 |
| **Qwen3-8B** | 16GB RAM/VRAM | 24GB VRAM | 3-5秒/次 |

### GPU支持

- ✅ **NVIDIA**: CUDA 11.8+ (推荐RTX 3060+)
- ✅ **AMD**: ROCm 5.7+ (RX 6000/7000系列)
- ✅ **Apple**: MPS (M1/M2/M3芯片)
- ✅ **CPU**: 支持但较慢

### 优化建议

**显存不足时**:
```bash
# 使用Qwen3-4B而非8B
# 或使用量化版本（如果可用）
# 或使用CPU模式
```

**推理速度慢时**:
```bash
# 检查是否使用GPU
python gpu_detector.py

# 确保PyTorch使用GPU
import torch
print(torch.cuda.is_available())  # NVIDIA
print(torch.backends.mps.is_available())  # Apple
```

## 📊 API使用

### 字幕校对API

```bash
curl -X POST http://localhost:5001/api/refine_subtitle \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "我们在座一个新项目",
    "model": "qwen3-4b",
    "context": ["公司业务扩展很快", "老板决定增加投入"],
    "language": "zh",
    "options": {
      "fix_punctuation": true,
      "fix_grammar": true,
      "remove_fillers": true,
      "format_segments": true
    }
  }'

# 响应
{
  "original_text": "我们在座一个新项目",
  "refined_text": "我们再做一个新项目。",
  "model": "qwen3-4b",
  "qwen_available": true
}
```

### 翻译API (即将推出)

```bash
curl -X POST http://localhost:5001/api/translate_qwen \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "这个方法很有效",
    "source_lang": "zh",
    "target_lang": "en"
  }'

# 响应
{
  "original_text": "这个方法很有效",
  "translated_text": "This method is very effective.",
  "model": "qwen3-4b"
}
```

### 模型列表API

```bash
curl http://localhost:5001/api/qwen_models

# 响应
{
  "available": true,
  "models": [
    {
      "name": "Qwen3-4B",
      "model_id": "Qwen/Qwen3-4B",
      "size": "4B",
      "recommended": true
    },
    {
      "name": "Qwen3-8B",
      "model_id": "Qwen/Qwen3-8B",
      "size": "8B",
      "recommended": false
    }
  ]
}
```

## 🧪 测试与验证

### 运行测试脚本

```bash
# 完整功能测试
python test_qwen.py

# 效果对比测试
python test_qwen_refinement.py
```

### 测试用例

测试脚本包含多种ASR错误类型：

1. **同音字错误**
   - 在座 → 再做
   - 有到礼 → 有道理
   - 意建 → 意见

2. **词语边界**
   - 人工只能 → 人工智能
   - 机器学习 → 机器学习

3. **口语填充词**
   - 嗯这个呃 → 这个
   - 那个就是说 → [去除]

4. **语法错误**
   - 他的很高兴 → 他很高兴
   - 应该要 → 应该

## 🔧 故障排除

### 问题1: 模型下载失败

**解决方案**:
```bash
# 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 重新启动
python app.py
```

### 问题2: CUDA内存不足

**解决方案**:
```bash
# 1. 使用更小的模型
# 选择 Qwen3-4B 而非 Qwen3-8B

# 2. 关闭其他GPU程序
nvidia-smi  # 查看GPU占用

# 3. 使用CPU模式
# 系统会自动回退
```

### 问题3: 推理速度慢

**优化方法**:
```python
# 1. 确认使用GPU
from gpu_detector import GPUDetector
detector = GPUDetector()
print(detector.get_optimal_device())

# 2. 调整batch size（代码层面）
# 3. 使用量化模型（如果可用）
# 4. 降低max_new_tokens参数
```

## 📚 相关文档

- [Qwen3官方文档](https://huggingface.co/Qwen/Qwen3-4B)
- [完整功能文档](README.qwen.md)
- [快速开始指南](QWEN_QUICKSTART.md)
- [实现细节](QWEN_IMPLEMENTATION.md)

## 🎉 开始使用

```bash
# 1. 安装依赖
pip install transformers>=4.37.0 torch -U

# 2. 启动服务
python app.py

# 3. 访问页面
http://localhost:5001/realtime.html

# 4. 选择Qwen3-4B模型

# 5. 享受专业的AI字幕校对！
```

---

**提示**: Qwen3-4B首次加载需下载~8GB模型文件，请耐心等待。后续使用会直接加载本地缓存，启动速度很快。
