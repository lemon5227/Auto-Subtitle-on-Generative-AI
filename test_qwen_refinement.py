#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen字幕校对效果演示和测试脚本
展示Qwen2.5-3B和Qwen2.5-7B在字幕校对上的实际效果
"""

import sys
import os

# 测试用例 - 包含各种常见的ASR错误
TEST_CASES = [
    {
        "category": "同音字错误",
        "examples": [
            {
                "context": ["我们公司最近业务扩展很快", "老板决定增加投入"],
                "input": "我们在座一个新项目",
                "expected": "我们再做一个新项目"
            },
            {
                "context": ["这个算法的设计很巧妙", "大家都很认可"],
                "input": "他说的话有到礼",
                "expected": "他说的话有道理"
            },
            {
                "context": ["这次会议讨论了很多技术问题"],
                "input": "大家的意建都很中肯",
                "expected": "大家的意见都很中肯"
            }
        ]
    },
    {
        "category": "词语边界错误",
        "examples": [
            {
                "context": ["我们在研究深度学习"],
                "input": "机器学习的效果很好",
                "expected": "机器学习的效果很好"
            },
            {
                "context": ["这个功能需要优化"],
                "input": "人工只能技术在进步",
                "expected": "人工智能技术在进步"
            }
        ]
    },
    {
        "category": "口语填充词",
        "examples": [
            {
                "context": [],
                "input": "嗯这个呃方法很有效",
                "expected": "这个方法很有效"
            },
            {
                "context": [],
                "input": "那个就是说我们需要改进",
                "expected": "我们需要改进"
            }
        ]
    },
    {
        "category": "语法错误",
        "examples": [
            {
                "context": ["项目进展顺利"],
                "input": "他的很高兴完成了任务",
                "expected": "他很高兴完成了任务"
            },
            {
                "context": ["大家都在努力工作"],
                "input": "我们应该要更加努力",
                "expected": "我们应该更加努力"
            }
        ]
    },
    {
        "category": "技术术语",
        "examples": [
            {
                "context": ["我们使用了最新的AI技术"],
                "input": "深度学习模型需要大量数据",
                "expected": "深度学习模型需要大量数据"
            },
            {
                "context": ["这是一个Python项目"],
                "input": "我们用了菲森语言开发",
                "expected": "我们用了Python语言开发"
            }
        ]
    }
]

def test_with_api(text, context=None, model='qwen-3b', language='zh'):
    """使用API测试字幕校对"""
    try:
        import requests
        
        response = requests.post(
            'http://localhost:5001/api/refine_subtitle',
            json={
                'text': text,
                'model': model,
                'context': context or [],
                'language': language,
                'options': {
                    'fix_punctuation': True,
                    'fix_grammar': True,
                    'remove_fillers': True,
                    'format_segments': True
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('refined_text', text)
        else:
            return None
    except Exception as e:
        print(f"API调用失败: {e}")
        return None

def test_local_refinement(text):
    """测试本地规则优化"""
    import re
    
    refined = text
    
    # 去除口语词
    fillers = ['嗯', '啊', '呃', '那个', '这个', '就是说', '然后']
    for filler in fillers:
        refined = refined.replace(filler, '')
    
    # 修正空格
    refined = re.sub(r'\s+', ' ', refined).strip()
    
    # 添加标点
    if refined and not refined[-1] in '.!?。！？':
        refined += '。'
    
    return refined

def calculate_similarity(text1, text2):
    """计算两个文本的相似度（简单字符重叠率）"""
    if not text1 or not text2:
        return 0.0
    
    # 移除空格和标点
    import re
    t1 = re.sub(r'[^\w]', '', text1)
    t2 = re.sub(r'[^\w]', '', text2)
    
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
    
    # 计算字符级别的重叠
    matches = sum(1 for c1, c2 in zip(t1, t2) if c1 == c2)
    return matches / max(len(t1), len(t2))

def print_comparison(category, example, local_result, qwen_result, expected):
    """打印对比结果"""
    print(f"\n{'='*70}")
    print(f"分类: {category}")
    print(f"{'='*70}")
    
    if example['context']:
        print(f"上下文:")
        for i, ctx in enumerate(example['context'], 1):
            print(f"  {i}. {ctx}")
        print()
    
    print(f"原始输入: {example['input']}")
    print(f"期望输出: {expected}")
    print(f"-" * 70)
    print(f"本地规则: {local_result}")
    
    if qwen_result:
        print(f"Qwen优化: {qwen_result}")
        
        # 计算相似度
        local_sim = calculate_similarity(local_result, expected)
        qwen_sim = calculate_similarity(qwen_result, expected)
        
        print(f"-" * 70)
        print(f"与期望相似度:")
        print(f"  本地规则: {local_sim*100:.1f}%")
        print(f"  Qwen优化: {qwen_sim*100:.1f}%")
        
        if qwen_sim > local_sim:
            print(f"✅ Qwen表现更好 (提升 {(qwen_sim-local_sim)*100:.1f}%)")
        elif qwen_sim == local_sim:
            print(f"⚖️ 两者表现相当")
        else:
            print(f"⚠️ 本地规则表现更好")
    else:
        print(f"Qwen优化: [API不可用]")

def main():
    """主测试流程"""
    print("\n" + "🧪" * 35)
    print("   Qwen2.5 字幕校对效果演示")
    print("🧪" * 35)
    
    # 检查服务状态
    print("\n检查服务状态...")
    try:
        import requests
        response = requests.get('http://localhost:5001/api/qwen_models', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('available'):
                print("✅ Qwen服务可用")
                print(f"   支持的模型: {len(data.get('models', []))}个")
            else:
                print("⚠️ Qwen不可用，只能测试本地规则")
                print("   请确保已安装: pip install transformers>=4.37.0")
        else:
            print("❌ 服务未响应")
            return
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        print("\n请先启动服务:")
        print("  python app.py")
        return
    
    # 选择测试模型
    print("\n选择测试模型:")
    print("  1. Qwen2.5-3B (推荐，平衡)")
    print("  2. Qwen2.5-7B (高质量)")
    print("  3. 两者都测试")
    
    choice = input("\n请选择 (1/2/3) [默认1]: ").strip() or "1"
    
    models_to_test = []
    if choice == "1":
        models_to_test = [('qwen-3b', 'Qwen2.5-3B')]
    elif choice == "2":
        models_to_test = [('qwen-7b', 'Qwen2.5-7B')]
    elif choice == "3":
        models_to_test = [('qwen-3b', 'Qwen2.5-3B'), ('qwen-7b', 'Qwen2.5-7B')]
    else:
        print("无效选择，使用默认Qwen2.5-3B")
        models_to_test = [('qwen-3b', 'Qwen2.5-3B')]
    
    # 运行测试
    total_tests = 0
    qwen_better = 0
    local_better = 0
    equal = 0
    
    for model_id, model_name in models_to_test:
        print(f"\n{'#'*70}")
        print(f"# 使用模型: {model_name}")
        print(f"{'#'*70}")
        
        for category_data in TEST_CASES:
            category = category_data['category']
            
            for example in category_data['examples']:
                total_tests += 1
                
                # 本地规则优化
                local_result = test_local_refinement(example['input'])
                
                # Qwen优化
                qwen_result = test_with_api(
                    example['input'],
                    context=example['context'],
                    model=model_id,
                    language='zh'
                )
                
                # 打印对比
                print_comparison(
                    category,
                    example,
                    local_result,
                    qwen_result,
                    example['expected']
                )
                
                # 统计
                if qwen_result:
                    local_sim = calculate_similarity(local_result, example['expected'])
                    qwen_sim = calculate_similarity(qwen_result, example['expected'])
                    
                    if qwen_sim > local_sim + 0.1:  # 阈值0.1避免微小差异
                        qwen_better += 1
                    elif local_sim > qwen_sim + 0.1:
                        local_better += 1
                    else:
                        equal += 1
                
                # 暂停以便查看
                if total_tests % 3 == 0:
                    input("\n按回车继续下一组测试...")
    
    # 打印总结
    print(f"\n{'='*70}")
    print("📊 测试总结")
    print(f"{'='*70}")
    print(f"总测试数: {total_tests}")
    print(f"Qwen表现更好: {qwen_better} ({qwen_better/total_tests*100:.1f}%)")
    print(f"本地规则更好: {local_better} ({local_better/total_tests*100:.1f}%)")
    print(f"表现相当: {equal} ({equal/total_tests*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("💡 使用建议")
    print(f"{'='*70}")
    
    if qwen_better > total_tests * 0.6:
        print("✅ Qwen在大多数场景下表现优秀")
        print("   推荐在重要场合使用Qwen优化")
        print("   日常使用可根据性能需求选择本地规则或Qwen")
    elif qwen_better > total_tests * 0.3:
        print("⚖️ Qwen在部分场景下有明显优势")
        print("   建议根据具体场景选择:")
        print("   - 同音字、专业术语 → 使用Qwen")
        print("   - 简单口语词去除 → 本地规则即可")
    else:
        print("⚠️ Qwen优势不明显")
        print("   可能原因:")
        print("   - 测试用例较简单")
        print("   - 模型参数需要调整")
        print("   - prompt需要进一步优化")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
