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
        self.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        whisper_dir = './models/whisper'
        # 获取权重数据类型
        self.weight_dtype = parent.unet.model.dtype
        
        # 加载Whisper模型
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)  # 禁用梯度计算

    def run_step_origin(self):
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
        for _ in range(self.batch_size*2): # 默认16个视频帧，32个音频帧
            audio_frame,type,eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type,eventpoint)) 
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]  # 合成完整音频流 (32个音频帧 = 640ms的音频数据)
        whisper_feature = self.audio_processor.audio2feat(inputs)
        # for feature in whisper_feature:
        #     self.audio_feats.append(feature)        
        #print(f"processing audio costs {(time.time() - start_time) * 1000}ms, inputs shape:{inputs.shape} whisper_feature len:{len(whisper_feature)}")
        whisper_chunks = self.audio_processor.feature2chunks(
            feature_array=whisper_feature,
            fps=self.fps/2,
            batch_size=self.batch_size,
            start=self.stride_left_size/2 )
        #print(f"whisper_chunks len:{len(whisper_chunks)},self.audio_feats len:{len(self.audio_feats)},self.output_queue len:{self.output_queue.qsize()}")
        #self.audio_feats = self.audio_feats[-(self.stride_left_size + self.stride_right_size):]
        self.feat_queue.put(whisper_chunks)
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

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
        for _ in range(self.batch_size*2): # 默认16个视频帧，32个音频帧
            audio_frame,type,eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type,eventpoint)) 
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]  # 合成完整音频流 (32个音频帧 = 640ms的音频数据)
        # whisper_feature = self.audio_processor.audio2feat(inputs)
        # whisper_chunks = self.audio_processor.feature2chunks(
        #     feature_array=whisper_feature,
        #     fps=self.fps/2,
        #     batch_size=self.batch_size,
        #     start=self.stride_left_size/2 )

        whisper_input_features, librosa_length = self.audio_processor.get_audio_stream_feature(inputs)  # 获取音频特征 

        print(f"whisper_input_features: {whisper_input_features.__len__}")
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
        print(f"whisper_chunks: {whisper_chunks.__len__()}")

        self.feat_queue.put(whisper_chunks)
        print(f"feat_queue put")
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

        print(f"run_step done")