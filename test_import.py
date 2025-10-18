#!/usr/bin/env python3
"""
测试循环导入问题的脚本
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入是否成功"""
    try:
        print("测试导入 museasr...")
        from museasr import MuseASR
        print("✓ museasr 导入成功")
        
        print("测试导入 musereal...")
        from musereal import MuseReal
        print("✓ musereal 导入成功")
        
        print("测试同时导入...")
        from museasr import MuseASR
        from musereal import MuseReal
        print("✓ 同时导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_circular_import():
    """测试循环导入问题"""
    try:
        print("\n测试循环导入...")
        
        # 先导入 musereal
        import musereal
        print("✓ musereal 模块导入成功")
        
        # 再导入 museasr
        import museasr
        print("✓ museasr 模块导入成功")
        
        # 检查是否可以访问类
        print(f"✓ MuseReal 类: {musereal.MuseReal}")
        print(f"✓ MuseASR 类: {museasr.MuseASR}")
        
        return True
        
    except Exception as e:
        print(f"✗ 循环导入测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("测试循环导入修复")
    print("=" * 50)
    
    # 测试基本导入
    import_test_passed = test_imports()
    
    # 测试循环导入
    circular_test_passed = test_circular_import()
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    print(f"基本导入测试: {'通过' if import_test_passed else '失败'}")
    print(f"循环导入测试: {'通过' if circular_test_passed else '失败'}")
    
    if import_test_passed and circular_test_passed:
        print("\n🎉 循环导入问题已解决！")
        return 0
    else:
        print("\n❌ 仍有问题需要解决")
        return 1

if __name__ == "__main__":
    exit(main())

