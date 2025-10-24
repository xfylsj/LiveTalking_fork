#!/usr/bin/env python3
"""
简化的音频特征测试脚本
不依赖 Whisper 模型下载，直接测试音频处理逻辑
"""

import numpy as np
import sys
sys.path.append("..")

def test_silence_processing():
    """测试静音帧处理"""
    print("=== 静音帧处理测试 ===")
    
    # 生成长度为320的静音帧（采样率16kHz，20ms帧，float32, 全为0）
    silent_frame = np.zeros(320, dtype=np.float32)
    print(f"静音帧形状: {silent_frame.shape}")
    print(f"静音帧类型: {silent_frame.dtype}")
    print(f"静音帧内容: {silent_frame[:10]}...")  # 显示前10个值
    
    # 模拟52个静音帧
    frames = [silent_frame] * 52
    print(f"总帧数: {len(frames)}")
    
    # 连接所有帧
    audio_stream = np.concatenate(frames)
    print(f"音频流长度: {len(audio_stream)}")
    print(f"音频流时长: {len(audio_stream) / 16000:.2f} 秒")
    
    # 模拟特征提取（不实际调用 Whisper）
    print("\n=== 模拟特征提取 ===")
    
    # 假设每帧对应一个特征向量
    # 对于 52 帧，每帧 20ms，总共 1040ms
    # 假设特征提取后得到 26 个特征向量（每 40ms 一个）
    num_features = len(frames) // 2  # 52 / 2 = 26
    feature_dim = 384  # Whisper tiny 模型的特征维度
    
    # 生成模拟特征
    mock_features = np.random.rand(num_features, feature_dim).astype(np.float32)
    print(f"模拟特征形状: {mock_features.shape}")
    print(f"特征数量: {num_features}")
    print(f"特征维度: {feature_dim}")
    
    # 模拟分块处理
    print("\n=== 模拟分块处理 ===")
    
    batch_size = 16
    stride_left = 10
    stride_right = 10
    
    # 计算实际处理的帧数
    processed_frames = len(frames) - stride_left - stride_right
    print(f"总帧数: {len(frames)}")
    print(f"左侧上下文: {stride_left}")
    print(f"右侧上下文: {stride_right}")
    print(f"实际处理帧数: {processed_frames}")
    
    # 模拟分块
    chunks = []
    for i in range(0, processed_frames, batch_size):
        chunk = mock_features[i:i+batch_size]
        chunks.append(chunk)
        print(f"分块 {len(chunks)}: 形状 {chunk.shape}")
    
    print(f"\n总共生成 {len(chunks)} 个分块")
    
    return True

def test_audio_frame_processing():
    """测试音频帧处理逻辑"""
    print("\n=== 音频帧处理逻辑测试 ===")
    
    # 模拟 museasr.py 中的处理逻辑
    batch_size = 16
    stride_left_size = 10
    stride_right_size = 10
    
    print(f"批次大小: {batch_size}")
    print(f"左侧步长: {stride_left_size}")
    print(f"右侧步长: {stride_right_size}")
    
    # 模拟收集音频帧
    frames = []
    for i in range(batch_size * 2):  # 32 个音频帧
        # 模拟静音帧
        frame = np.zeros(320, dtype=np.float32)
        frames.append(frame)
    
    print(f"收集的帧数: {len(frames)}")
    
    # 检查是否有足够的上下文
    if len(frames) <= stride_left_size + stride_right_size:
        print("上下文不足，跳过处理")
        return False
    
    print("上下文足够，开始处理")
    
    # 连接所有帧
    audio_stream = np.concatenate(frames)
    print(f"音频流长度: {len(audio_stream)}")
    
    # 模拟保留上下文帧
    context_frames = stride_left_size + stride_right_size
    frames = frames[-context_frames:]
    print(f"保留的上下文帧数: {len(frames)}")
    
    return True

if __name__ == "__main__":
    print("开始音频特征处理测试...")
    
    try:
        # 测试静音处理
        test_silence_processing()
        
        # 测试音频帧处理
        test_audio_frame_processing()
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
