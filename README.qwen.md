# Qwen智能字幕校对功能指南

## 功能概述

本项目已集成阿里云**Qwen大语言模型**，为实时语音转录提供智能字幕校对功能。Qwen可以根据上下文自动发现并修正语音识别中的错误，大幅提升字幕质量。

## 核心特性

### 🎯 智能纠错
- **同音字修正**：自动识别并修正同音字错误（如"在座"→"再做"）
- **词语边界识别**：根据上下文正确分词
- **语法优化**：自动修正语法错误和不通顺的表达
- **语义理解**：基于前后文推断说话人的真实意图

### 🚀 本地部署
- **完全离线运行**：所有模型在本地运行，无需联网
- **隐私保护**：字幕数据不会上传到外部服务器
- **支持多种规格**：
  - Qwen-1.5B：轻量级，适合低配置设备
  - Qwen-3B：**推荐**，性能与质量平衡最佳
  - Qwen-7B：高质量，需要更多显存

## 快速开始

### 1. 安装依赖

首先确保已安装基础环境（Python 3.8+、PyTorch等），然后安装transformers库：

```bash
pip install transformers>=4.37.0
```

### 2. 模型下载

Qwen模型会在首次使用时自动从HuggingFace下载：

```python
# 自动下载到 ~/.cache/huggingface/hub/
Qwen/Qwen2.5-3B-Instruct  # 推荐，约6GB
Qwen/Qwen2.5-7B-Instruct  # 高质量，约14GB
Qwen/Qwen2.5-1.5B-Instruct  # 轻量，约3GB
```

**国内加速下载（可选）：**

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

手动下载（如果自动下载失败）：

```bash
# 安装下载工具
pip install huggingface-cli

# 下载Qwen-3B（推荐）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/qwen-3b
```

### 3. 启动服务

```bash
python app.py
```

服务启动后会自动检测Qwen模型是否可用：
- ✅ **Qwen LLM support available** - 模型可用
- ⚠️ **Qwen LLM not available** - 模型未安装，仅使用规则优化

### 4. 使用字幕校对

1. 访问实时转录页面：http://localhost:5001/realtime.html
2. 点击"高级设置"展开功能面板
3. 勾选"✨ 启用字幕优化"
4. 选择模型：
   - **本地规则**：基于正则表达式，快速但功能有限
   - **Qwen-3B**：推荐，速度快且质量高
   - **Qwen-7B**：最高质量，需要更多显存
5. 配置优化选项：
   - ✅ 标点符号修正
   - ✅ 语法优化（使用Qwen模型）
   - ✅ 去除口语词（嗯、啊等）
   - ✅ 格式优化

## 技术原理

### 工作流程

```
语音输入 → Whisper转录 → 实时字幕 → Qwen智能校对 → 优化字幕
                                    ↑
                                上下文分析
```

### 上下文感知

Qwen模型会自动获取最近3条字幕作为上下文，实现：

1. **连贯性检查**：确保前后文逻辑一致
2. **主题理解**：根据对话主题推断正确词汇
3. **说话习惯学习**：适应说话人的表达风格

### 示例对比

| 原始识别 | 本地规则优化 | Qwen智能优化 |
|---------|-------------|-------------|
| 我们在座一个新的项目 | 我们在座一个新的项目。 | 我们再做一个新的项目。 |
| 这个呃方法很有笑 | 这个方法很有笑。 | 这个方法很有效。 |
| 他说的话有到礼 | 他说的话有到礼。 | 他说的话有道理。 |

## API使用

### 字幕优化API

**端点：** `POST /api/refine_subtitle`

**请求参数：**
```json
{
  "text": "需要优化的字幕文本",
  "model": "qwen-3b",
  "context": ["前一条字幕", "前两条字幕"],
  "language": "zh",
  "options": {
    "fix_punctuation": true,
    "fix_grammar": true,
    "remove_fillers": true,
    "format_segments": true
  }
}
```

**响应示例：**
```json
{
  "refined_text": "优化后的字幕",
  "original_text": "原始字幕",
  "model": "qwen-3b",
  "qwen_available": true
}
```

### 模型列表API

**端点：** `GET /api/qwen_models`

**响应示例：**
```json
{
  "available": true,
  "models": [
    {
      "name": "Qwen2.5-3B-Instruct",
      "model_id": "Qwen/Qwen2.5-3B-Instruct",
      "size": "3B",
      "recommended": true
    },
    ...
  ]
}
```

## 性能优化

### GPU加速

系统会自动检测并使用最佳计算设备：

```python
from gpu_detector import GPUDetector

detector = GPUDetector()
device = detector.get_optimal_device()
# CUDA (NVIDIA) / ROCm (AMD) / MPS (Apple) / CPU
```

### 内存管理

- **FP16精度**：GPU上使用半精度浮点，节省50%显存
- **模型缓存**：首次加载后常驻内存，避免重复加载
- **按需加载**：只有启用Qwen优化时才加载模型

### 推荐配置

| 模型 | 最低显存 | 推荐显存 | 推理速度 | 质量评分 |
|-----|---------|---------|---------|---------|
| Qwen-1.5B | 4GB | 6GB | 极快 | ⭐⭐⭐ |
| Qwen-3B | 6GB | 8GB | 快 | ⭐⭐⭐⭐ |
| Qwen-7B | 12GB | 16GB | 中等 | ⭐⭐⭐⭐⭐ |

**注意：** CPU模式下建议使用Qwen-1.5B或3B，7B模型可能较慢。

## 故障排除

### 问题1：模型下载失败

**症状：** `⚠️ Qwen LLM not available`

**解决方案：**
```bash
# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 手动下载
huggingface-cli download Qwen/Qwen2.5-3B-Instruct

# 或使用git-lfs
git lfs install
git clone https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct
```

### 问题2：显存不足

**症状：** `CUDA out of memory`

**解决方案：**
1. 切换到更小的模型（3B → 1.5B）
2. 关闭其他占用显存的程序
3. 使用CPU模式（速度较慢）：
```python
# 在app.py中设置
DEVICE = 'cpu'
```

### 问题3：推理速度慢

**症状：** 字幕校对延迟明显

**优化建议：**
1. 确保使用GPU加速
2. 降低temperature参数（已默认0.3）
3. 减少max_new_tokens（已默认256）
4. 考虑使用更小的模型
5. 只在关键场景启用Qwen优化

### 问题4：校对质量不佳

**调优建议：**
1. 提供更多上下文（当前默认3条）
2. 调整prompt模板（在`refine_subtitle_with_qwen`函数中）
3. 尝试更大的模型（7B质量最高）
4. 针对特定领域添加专业术语提示

## 最佳实践

### 1. 场景选择

**推荐使用Qwen的场景：**
- 专业会议录音（术语较多）
- 教育培训内容
- 访谈对话
- 技术演讲

**可使用本地规则的场景：**
- 简单日常对话
- 新闻播报（发音标准）
- 朗读文本

### 2. 配置建议

**高质量场景：**
```javascript
{
  model: "qwen-7b",
  options: {
    fix_punctuation: true,
    fix_grammar: true,
    remove_fillers: true,
    format_segments: true
  }
}
```

**实时性优先场景：**
```javascript
{
  model: "local",  // 或 qwen-1.5b
  options: {
    fix_punctuation: true,
    remove_fillers: true
  }
}
```

### 3. 批量处理

对于已录制的音频/视频，建议：
1. 先完成转录
2. 保存原始字幕
3. 批量调用Qwen优化
4. 人工复核关键内容

## 扩展开发

### 自定义Prompt

修改`app.py`中的`refine_subtitle_with_qwen`函数：

```python
system_prompt = """你是一个专业的字幕校对助手。
特别注意：
1. 本次对话主题是：【科技产品评测】
2. 常见术语：AI、机器学习、深度学习、GPU等
3. 说话人习惯使用口语化表达
..."""
```

### 集成其他模型

参考现有实现，可集成：
- GPT系列（需要API key）
- ChatGLM（清华开源）
- Baichuan（百川智能）

```python
def refine_with_custom_model(text, context):
    # 你的模型实现
    pass
```

## 更新日志

### v1.0 (2024-01)
- ✅ 集成Qwen 2.5系列模型
- ✅ 支持上下文感知校对
- ✅ 自动GPU检测和加速
- ✅ 前端UI集成
- ✅ 本地规则降级方案

## 参考资源

- [Qwen官方仓库](https://github.com/QwenLM/Qwen)
- [Transformers文档](https://huggingface.co/docs/transformers)
- [项目主页](README.md)

## 技术支持

遇到问题？
1. 查看控制台日志
2. 检查`server.log`
3. 提交Issue到GitHub

---

**提示：** Qwen模型首次加载需要下载6-14GB文件，请确保网络畅通和存储空间充足。
