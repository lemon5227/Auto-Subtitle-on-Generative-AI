#!/bin/bash
# -*- coding: utf-8 -*-
# 快速下载Qwen模型脚本

echo "=================================="
echo "  Qwen模型下载工具"
echo "=================================="
echo ""

# 检测Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    echo "请先安装Python 3.8+"
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "使用Python: $PYTHON_CMD"
echo ""

# 显示菜单
echo "请选择要下载的模型:"
echo ""
echo "1) 实时翻译模型 (Qwen3-0.6B + 1.7B, ~4GB, 推荐)"
echo "2) 字幕优化模型 (Qwen3-4B, ~8GB)"
echo "3) 所有模型 (Qwen3-0.6B/1.7B/4B/8B, ~32GB)"
echo "4) 仅超轻量模型 (Qwen3-0.6B, ~2GB, 低配设备)"
echo "5) 退出"
echo ""

read -p "请输入选项 [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "📥 下载实时翻译模型 (Qwen3-0.6B + 1.7B)..."
        $PYTHON_CMD test_qwen.py --download --model realtime --skip-tests
        ;;
    2)
        echo ""
        echo "📥 下载字幕优化模型 (Qwen3-4B)..."
        $PYTHON_CMD test_qwen.py --download --model refinement --skip-tests
        ;;
    3)
        echo ""
        echo "📥 下载所有模型 (可能需要30分钟+)..."
        read -p "确认下载所有模型? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            $PYTHON_CMD test_qwen.py --download --model all --skip-tests
        else
            echo "已取消"
            exit 0
        fi
        ;;
    4)
        echo ""
        echo "📥 下载超轻量模型 (Qwen3-0.6B)..."
        # 临时修改test_qwen.py只下载0.6B
        $PYTHON_CMD -c "
from test_qwen import download_qwen_model
import sys

device = 'cpu'
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except:
    pass

print(f'设备: {device}')
success = download_qwen_model('Qwen/Qwen3-0.6B', device=device, use_fp16=False)
sys.exit(0 if success else 1)
"
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ 下载完成！"
    echo "=================================="
    echo ""
    echo "下一步:"
    echo "1. 启动服务: python app.py"
    echo "2. 访问页面: http://localhost:5001/realtime.html"
    echo "3. 配置翻译功能并开始使用"
    echo ""
else
    echo ""
    echo "=================================="
    echo "❌ 下载失败"
    echo "=================================="
    echo ""
    echo "常见问题:"
    echo "1. 网络问题 - 可以设置镜像:"
    echo "   export HF_ENDPOINT=https://hf-mirror.com"
    echo "   ./download_models.sh"
    echo ""
    echo "2. 磁盘空间不足 - 清理~/.cache/huggingface"
    echo ""
    echo "3. 其他问题 - 查看 TRANSLATION_OPTIMIZATION.md"
    echo ""
    exit 1
fi
