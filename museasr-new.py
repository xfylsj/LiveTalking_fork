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

import sys
import time
import numpy as np

import queue
from queue import Queue
#import multiprocessing as mp
from baseasr import BaseASR
from musetalk.utils.audio_processor import AudioProcessor
# from musetalk.whisper.audio2feature import Audio2Feature
import torch
from transformers import WhisperModel

class MuseASR(BaseASR):
    def __init__(self, opt, parent, audio_processor: AudioProcessor):
        super().__init__(opt,parent)
        self.audio_processor = audio_processor

                # 设置计算设备
        # self.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        self.device = parent.unet.model.device
        whisper_dir = './models/whisper'
        # 获取权重数据类型
        self.weight_dtype = parent.unet.model.dtype
        
        # 加载Whisper模型
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)  # 禁用梯度计算

    def run_step(self):
        """
        运行一步音频处理，提取音频特征
        
        处理流程：
        1. 收集音频帧
        2. 提取音频特征
        3. 将特征分块
        4. 将处理后的特征放入队列
        """
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        self.frames = []
        print(f'self.batch_size = {self.batch_size}')
        for _ in range(self.batch_size*2): # 默认16个视频帧，32个音频帧
            audio_frame,type,eventpoint = self.get_audio_frame()    # 没有声音的时候会得到“静音帧”, 长度320个0， type = 1
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type,eventpoint)) 
        
        # if len(self.frames) <= self.stride_left_size + self.stride_right_size:
        #     return


        inputs = np.concatenate(self.frames) # [N * chunk]  # 合成完整音频流 (32个音频帧 = 640ms的音频数据)


        whisper_input_features, librosa_length = self.audio_processor.get_audio_stream_feature(inputs, weight_dtype=self.weight_dtype)  # 获取音频特征 


        
        whisper_chunks = self.audio_processor.get_whisper_chunk(  # 获取Whisper模型的分块
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=self.fps/2,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
            # audio_padding_length_left=self.stride_left_size,
            # audio_padding_length_right=self.stride_right_size,
        )

        # 转换为cpu数据放入队列
        self.feat_queue.put(whisper_chunks.cpu().numpy())
    
        # discard the old part to save memory
        # self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
        """
        保留 20 个帧是为了：
            提供音频上下文：确保 Whisper 模型有足够的上下文进行准确识别
            保证音频连续性：避免音频边界效应
            优化特征质量：有上下文的特征提取更稳定
            内存管理：只保留必要的上下文，避免内存无限增长
            这是音频实时处理中的标准做法，确保模型能够获得最佳的音频特征质量。
        """
