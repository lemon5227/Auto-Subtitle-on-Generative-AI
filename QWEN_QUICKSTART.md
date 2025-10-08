# Qwen智能字幕校对 - 快速开始指南

## 🎯 功能概览

本项目新增**Qwen大语言模型智能字幕校对**功能，可以：
- ✅ 自动修正语音识别中的同音字错误
- ✅ 根据上下文推断正确的词语
- ✅ 修复语法错误和不通顺的表达
- ✅ 去除口语填充词（嗯、啊、那个等）
- ✅ 完全本地运行，无需联网，保护隐私

## 🚀 快速上手（5分钟）

### 方式1：首次安装（完整版）

```bash
# 1. 安装/更新依赖
pip install transformers>=4.37.0 -U

# 2. 启动服务
python app.py

# 3. 访问页面
# 浏览器打开: http://localhost:5001/realtime.html
```

### 方式2：已有环境（升级版）

```bash
# 只需更新transformers
pip install transformers>=4.37.0 -U

# 重启服务即可
python app.py
```

## 💡 使用方法

### 在实时转录页面使用

1. **访问**: http://localhost:5001/realtime.html
2. **展开高级设置**: 点击"高级设置"按钮
3. **启用优化**: 勾选"✨ 启用字幕优化"
4. **选择模型**:
   - `本地规则` - 快速，基于正则表达式
   - `Qwen-1.5B` - 轻量，适合低配设备
   - `Qwen-3B` - **推荐**，速度与质量平衡
   - `Qwen-7B` - 最高质量，需要更多显存

5. **配置选项**:
   ```
   ✅ 标点符号修正 - 修正标点使用
   ✅ 语法优化 - 启用Qwen智能校对
   ✅ 去除口语词 - 去除嗯、啊等填充词
   ✅ 格式优化 - 优化文本格式
   ```

6. **开始录音**: 点击麦克风按钮，说话即可看到实时优化效果

### 效果对比

| 场景 | 原始识别 | 本地规则 | Qwen优化 |
|------|---------|---------|---------|
| 同音字 | 我们在座一个新项目 | 我们在座一个新项目。 | 我们再做一个新项目。 |
| 口语词 | 嗯这个呃方法很有效 | 方法很有效。 | 这个方法很有效。 |
| 语法 | 他说的话有到礼 | 他说的话有到礼。 | 他说的话有道理。 |

## 📦 模型下载

### 自动下载（推荐）

首次使用Qwen模型时会自动从HuggingFace下载：

```bash
# 启动服务后，选择Qwen模型
# 模型会自动下载到: ~/.cache/huggingface/hub/

# Qwen-1.5B: 约3GB
# Qwen-3B:   约6GB（推荐）
# Qwen-7B:   约14GB
```

### 国内加速（可选）

如果下载速度慢，可以使用镜像：

```bash
# 临时使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python app.py

# 或永久设置（添加到~/.bashrc或~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 手动下载（可选）

```bash
# 安装下载工具
pip install huggingface-hub

# 下载模型
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/qwen-3b

# 修改app.py指向本地路径
# model_id = "./models/qwen-3b"
```

## 🎮 硬件要求

### 推荐配置

| 模型 | 最低显存/内存 | 推荐配置 | 推理速度 | 质量 |
|------|--------------|---------|---------|------|
| Qwen-1.5B | 4GB | 6GB RAM/VRAM | 极快 | ⭐⭐⭐ |
| Qwen-3B | 6GB | 8GB RAM/VRAM | 快 | ⭐⭐⭐⭐ |
| Qwen-7B | 12GB | 16GB RAM/VRAM | 中等 | ⭐⭐⭐⭐⭐ |

### GPU加速

系统会自动检测GPU并启用加速：

```bash
# 检测GPU
python gpu_detector.py

# 示例输出
# 🎯 选择设备: cuda
# 📊 设备信息: 🟢 NVIDIA GPU (RTX 3060, 6GB) - 推荐Qwen-3B
```

支持的GPU：
- ✅ **NVIDIA**: CUDA加速（GTX 1060+、RTX系列）
- ✅ **AMD**: ROCm加速（RX 6000/7000系列）
- ✅ **Apple**: MPS加速（M1/M2/M3芯片）
- ✅ **CPU**: 多线程优化（所有平台）

## 🧪 测试功能

运行测试脚本验证安装：

```bash
python test_qwen.py
```

测试内容：
1. ✅ 库导入测试
2. ✅ GPU检测测试
3. ✅ 模型信息测试
4. ✅ 规则优化测试
5. ⚠️ 模型加载测试（可选）
6. ⚠️ API端点测试（可选）

## 📊 API使用

### 字幕优化API

```bash
# 基本用法
curl -X POST http://localhost:5001/api/refine_subtitle \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "嗯这个呃方法很有效",
    "model": "qwen-3b",
    "language": "zh",
    "options": {
      "fix_punctuation": true,
      "fix_grammar": true,
      "remove_fillers": true
    }
  }'

# 响应示例
{
  "original_text": "嗯这个呃方法很有效",
  "refined_text": "这个方法很有效。",
  "model": "qwen-3b",
  "qwen_available": true
}
```

### 带上下文的优化

```bash
curl -X POST http://localhost:5001/api/refine_subtitle \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "我们在座一个新项目",
    "model": "qwen-3b",
    "context": [
      "我们公司最近业务扩展很快",
      "老板决定增加投入"
    ],
    "language": "zh"
  }'

# Qwen会根据上下文推断"在座"应为"再做"
```

### 查询可用模型

```bash
curl http://localhost:5001/api/qwen_models

# 响应
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

## 🔧 故障排除

### 问题1: 模型未加载

**现象**: 控制台显示 `⚠️ Qwen LLM not available`

**解决**:
```bash
# 1. 检查transformers版本
pip show transformers
# 应该是 >= 4.37.0

# 2. 升级transformers
pip install transformers>=4.37.0 -U

# 3. 重启服务
python app.py
```

### 问题2: 下载速度慢或失败

**解决**:
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

### 问题3: 显存不足

**现象**: `CUDA out of memory`

**解决**:
```bash
# 1. 使用更小的模型
# 在realtime.html选择 Qwen-1.5B

# 2. 关闭其他占用显存的程序

# 3. 使用CPU模式（较慢但稳定）
# 系统会自动回退到CPU
```

### 问题4: 推理速度慢

**优化**:
1. 确保使用GPU加速
2. 选择更小的模型（1.5B或3B）
3. 只在需要高质量的场景启用Qwen
4. 其他场景使用"本地规则"优化

## 💡 使用建议

### 推荐场景

**使用Qwen优化**:
- ✅ 专业会议记录
- ✅ 教育培训内容
- ✅ 访谈对话
- ✅ 技术演讲
- ✅ 口音较重的音频

**使用本地规则**:
- ✅ 简单日常对话
- ✅ 新闻播报
- ✅ 朗读标准文本
- ✅ 实时性要求极高的场景

### 配置策略

**高质量优先**:
```javascript
{
  model: "qwen-7b",  // 最高质量
  options: {
    fix_punctuation: true,
    fix_grammar: true,
    remove_fillers: true,
    format_segments: true
  }
}
```

**速度优先**:
```javascript
{
  model: "local",  // 或 qwen-1.5b
  options: {
    fix_punctuation: true,
    remove_fillers: true
  }
}
```

## 📚 更多文档

- [完整功能文档](README.qwen.md)
- [AMD GPU支持](README.amd.md)
- [macOS部署](README.macOS.md)
- [Linux部署](README.linux.md)

## 🎉 开始使用

```bash
# 1. 启动服务
python app.py

# 2. 浏览器访问
http://localhost:5001/realtime.html

# 3. 开启"字幕优化"并选择Qwen模型

# 4. 享受智能字幕校对！
```

---

**提示**: 首次使用Qwen模型会下载6-14GB文件，请耐心等待。后续使用会直接加载本地缓存的模型，启动速度很快。
