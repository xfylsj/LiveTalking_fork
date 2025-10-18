###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import time
import numpy as np

import queue
from queue import Queue

import torch
#import multiprocessing as mp
from baseasr import BaseASR
# from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from musereal import MuseReal



class MuseASR(BaseASR):
    # def __init__(self, opt, parent, audio_processor: Audio2Feature):
    def __init__(self, opt, parent: "MuseReal", audio_processor: AudioProcessor):

        super().__init__(opt,parent)
        self.audio_processor = audio_processor

        # 设置计算设备
        self.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        whisper_dir = './models/whisper'
        # 获取权重数据类型
        self.weight_dtype = parent.unet.model.dtype
        
        # 加载Whisper模型
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)  # 禁用梯度计算


    def run_step_new(self):
        """ 运行一步音频处理，包括特征提取和分块处理
        
        处理流程：
        1. 收集音频帧
        2. 提取音频特征
        3. 将特征分块
        4. 将处理后的特征放入队列
        """
        # 记录开始时间，用于性能分析
        start_time = time.time()
        
        # 收集 batch_size*2 个音频帧
        # 每个视频帧对应两个音频帧，所以需要乘以2
        for _ in range(self.batch_size * 2):    # batch_size = 16
            # 获取音频帧、类型和事件点
            # type: 0-正常语音, 1-静音
            # eventpoint: 自定义事件同步点
            # 数据来源：ttsreal.py 的 stream_tts 方法
            audio_frame, type, eventpoint = self.get_audio_frame()
            # 将音频帧添加到帧列表
            self.frames.append(audio_frame)
            # 将音频帧放入输出队列，供其他组件使用
            self.output_queue.put((audio_frame, type, eventpoint))
        
        # 如果收集的帧数不够，直接返回
        # 需要足够的帧数来保证特征提取的上下文

        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        """
        # 将所有音频帧连接成一个连续的音频流

        audio_n_frame 是一个一维的 NumPy 数组
        形状: (N,)
        其中 N 是总采样点数
        数据类型: float32
        值范围: [-1.0, 1.0]
        例如：
        audio_n_frame = np.array([0.1, -0.2, 0.3, -0.1, ...])  # 形状: (16000,) 表示1秒的音频
        """
        audio_n_frame = np.concatenate(self.frames)

        
        ############### >>>>>>>>>>>>>
        # 提取音频特征
        whisper_input_features, librosa_length = self.audio_processor.get_audio_stream_feature(audio_n_frame)  # 获取音频特征 TODO: 需要修改成处理音频帧
        whisper_chunks = self.audio_processor.get_whisper_chunk(  # 获取Whisper模型的分块
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=self.fps/2,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        ###############  <<<<<<<<<<<<

        
        # 将处理后的特征块放入特征队列
        # 供后续的模型推理使用
        self.feat_queue.put(whisper_chunks)
        
        # 保留最新的帧，丢弃旧的帧
        # 只保留 stride_left_size + stride_right_size 个帧
        # 这样可以保证有足够的上下文，同时节省内存
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def run_step_origin(self):
        """ 运行一步音频处理，包括特征提取和分块处理
        
        处理流程：
        1. 收集音频帧
        2. 提取音频特征
        3. 将特征分块
        4. 将处理后的特征放入队列
        """
        # 记录开始时间，用于性能分析
        start_time = time.time()
        
        # 收集 batch_size*2 个音频帧
        # 每个视频帧对应两个音频帧，所以需要乘以2
        for _ in range(self.batch_size*2):
            # 获取音频帧、类型和事件点
            # type: 0-正常语音, 1-静音
            # eventpoint: 自定义事件同步点
            audio_frame, type, eventpoint = self.get_audio_frame()
            # 将音频帧添加到帧列表
            self.frames.append(audio_frame)
            # 将音频帧放入输出队列，供其他组件使用
            self.output_queue.put((audio_frame, type, eventpoint))
        
        # 如果收集的帧数不够，直接返回
        # 需要足够的帧数来保证特征提取的上下文
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        # 将所有音频帧连接成一个连续的音频流
        # [N * chunk] 表示 N 个音频块，每个块大小为 chunk
        inputs = np.concatenate(self.frames)

        # 使用音频处理器提取 Whisper 特征
        # 将原始音频转换为模型可用的特征表示
        whisper_feature = self.audio_processor.audio2feat(inputs)
        
        # 将特征数组分块处理
        # fps/2: 因为每个视频帧对应两个音频帧，所以特征帧率是视频帧率的一半
        # batch_size: 每批处理的帧数
        # start: 从 stride_left_size/2 开始，确保有足够的上下文
        whisper_chunks = self.audio_processor.feature2chunks(   # TODO: 对照分析不同 musetalk 中 realtime_inference.py 374行
            feature_array=whisper_feature,
            fps=self.fps/2,
            batch_size=self.batch_size,
            start=self.stride_left_size/2
        )
        
        # 将处理后的特征块放入特征队列
        # 供后续的模型推理使用
        self.feat_queue.put(whisper_chunks)
        
        # 保留最新的帧，丢弃旧的帧
        # 只保留 stride_left_size + stride_right_size 个帧
        # 这样可以保证有足够的上下文，同时节省内存
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def run_step_2(self):
        """ 运行一步音频处理，使用与 MuseTalk 一致的音频特征提取方法
        
        处理流程：
        1. 收集音频帧
        2. 使用 AudioProcessor.get_audio_stream_feature 提取音频特征
        3. 使用 AudioProcessor.get_whisper_chunk 生成 Whisper 分块
        4. 将处理后的特征放入队列
        """
        # 记录开始时间，用于性能分析
        start_time = time.time()
        
        # 收集 batch_size*2 个音频帧
        # 每个视频帧对应两个音频帧，所以需要乘以2
        for _ in range(self.batch_size * 2):
            # 获取音频帧、类型和事件点
            # type: 0-正常语音, 1-静音
            # eventpoint: 自定义事件同步点
            audio_frame, type, eventpoint = self.get_audio_frame()
            # 将音频帧添加到帧列表
            self.frames.append(audio_frame)
            # 将音频帧放入输出队列，供其他组件使用
            self.output_queue.put((audio_frame, type, eventpoint))
        
        # 如果收集的帧数不够，直接返回
        # 需要足够的帧数来保证特征提取的上下文
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        # 将所有音频帧连接成一个连续的音频流
        audio_n_frame = np.concatenate(self.frames)
        
        # 使用与 MuseTalk 一致的音频特征提取方法
        # 1. 使用 get_audio_stream_feature 提取音频特征
        whisper_input_features, librosa_length = self.audio_processor.get_audio_stream_feature(
            audio_n_frame, 
            weight_dtype=self.weight_dtype
        )
        
        # 2. 使用 get_whisper_chunk 生成 Whisper 分块
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=self.fps/2,  # 因为每个视频帧对应两个音频帧
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        
        # 将处理后的特征块放入特征队列
        # 供后续的模型推理使用
        self.feat_queue.put(whisper_chunks)
        
        # 保留最新的帧，丢弃旧的帧
        # 只保留 stride_left_size + stride_right_size 个帧
        # 这样可以保证有足够的上下文，同时节省内存
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]


