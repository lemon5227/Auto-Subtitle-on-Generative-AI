#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试模型管理API"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_list_models():
    """测试获取模型列表"""
    print("=" * 60)
    print("测试: GET /api/qwen_models/list")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/qwen_models/list", timeout=5)
        data = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"模型数量: {len(data.get('models', []))}")
        print(f"缓存目录: {data.get('cache_dir')}")
        
        print("\n模型列表:")
        for model in data.get('models', []):
            status = "✓ 已下载" if model.get('downloaded') else "✗ 未下载"
            print(f"  {status} {model['name']} ({model['size']}) - {model['best_for']}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_get_models():
    """测试获取支持的模型"""
    print("\n" + "=" * 60)
    print("测试: GET /api/qwen_models")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/qwen_models", timeout=5)
        data = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"Qwen可用: {data.get('available')}")
        print(f"支持模型数: {len(data.get('models', []))}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("\n🧪 开始测试模型管理API\n")
    print("⚠️  请确保服务已启动: python app.py\n")
    
    input("按回车继续...")
    
    results = []
    
    # 测试1: 获取支持的模型
    results.append(("获取支持模型", test_get_models()))
    
    # 测试2: 获取模型列表及状态
    results.append(("获取模型列表", test_list_models()))
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20} {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✅ 所有API测试通过！")
        print("\n现在可以访问: http://localhost:5001/realtime.html")
        print("点击高级设置，查看Qwen模型管理面板")
    else:
        print("\n⚠️  部分测试未通过")

if __name__ == "__main__":
    main()
