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

import math
import torch
import numpy as np

#from .utils import *
import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
#from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image,get_image_prepare_material,get_image_blending
from musetalk.utils.utils import load_all_model,load_diffusion_model,load_audio_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.whisper.audio2feature import Audio2Feature

import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

from tqdm import tqdm
from logger import logger

from transformers import WhisperModel


def load_model():
    # load model weights
    audio_processor, vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"))
    timesteps = torch.tensor([0], device=device)
    pe = pe.half()
    vae.vae = vae.vae.half()
    #vae.vae.share_memory()
    unet.model = unet.model.half()
    #unet.model.share_memory()
    return vae, unet, pe, timesteps, audio_processor

def load_avatar(avatar_id):
    #self.video_path = '' #video_path
    #self.bbox_shift = opt.bbox_shift
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path= f"{avatar_path}/latents.pt"
    video_out_path = f"{avatar_path}/vid_output/"
    mask_out_path =f"{avatar_path}/mask"
    mask_coords_path =f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"
    # self.avatar_info = {
    #     "avatar_id":self.avatar_id,
    #     "video_path":self.video_path,
    #     "bbox_shift":self.bbox_shift   
    # }

    input_latent_list_cycle = torch.load(latents_out_path)  #,weights_only=True
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle

@torch.no_grad()
def warm_up(batch_size,model):
    # 预热函数
    logger.info('warmup model...')
    vae, unet, pe, timesteps, audio_processor = model
    #batch_size = 16
    #timesteps = torch.tensor([0], device=unet.device)
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.uint8)
    latent_batch = torch.ones(batch_size, 8, 32, 32).to(unet.device)

    audio_feature_batch = torch.from_numpy(whisper_batch)
    audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(dtype=unet.model.dtype)
    pred_latents = unet.model(latent_batch,
                              timesteps,
                              encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 

@torch.no_grad()
def inference_origin(
        render_event,
        batch_size,
        input_latent_list_cycle,
        audio_feat_queue,
        audio_out_queue,
        res_frame_queue,
        vae,
        unet,
        pe,
        timesteps): #vae, unet, pe, timesteps
    """推理函数,用于生成音频驱动的视频帧。视频帧只有脸部大小，后期需要替换到指定图片的脸部位置
    Args:
        render_event: 渲染控制事件
        batch_size: 批处理大小
        input_latent_list_cycle: 循环使用的潜在向量列表
        audio_feat_queue: 音频特征队列
        audio_out_queue: 音频输出队列
        res_frame_queue: 结果帧队列
        vae: VAE模型
        unet: UNet模型
        pe: 位置编码
        timesteps: 时间步长
    """
    length = len(input_latent_list_cycle)
    index = 0
    count = 0  # 计数器,用于计算FPS
    counttime = 0  # 累计时间,用于计算FPS
    logger.info('start inference')

    while render_event.is_set():
        starttime = time.perf_counter()
        try:
            """
            获取音频特征。来源于 museasr.py 中的 feat_queue。
            其中的 whisper_chunks 在 169行获取，需要修改成 105 行的逻辑 
            """
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        # 获取音频帧并检查是否全为静音
        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type == 0:  # type=0表示非静音
                is_all_silence = False

        if is_all_silence:
            # 如果全是静音,直接输出空帧
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # 处理非静音帧
            t = time.perf_counter()
            
            # 准备输入数据
            whisper_batch = np.stack(whisper_chunks)
            latent_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)
            
            # 处理音频特征
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                       dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            # 使用UNet生成潜在向量
            pred_latents = unet.model(latent_batch, 
                                    timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
            
            # 使用VAE解码生成图像
            recon = vae.decode_latents(pred_latents)

            # 计算并输出FPS
            counttime += (time.perf_counter() - t)
            count += batch_size
            if count >= 100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count = 0
                counttime = 0

            # 输出生成的帧
            for i,res_frame in enumerate(recon):
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
            
    logger.info('musereal inference processor stop')


@torch.no_grad()
def inference(render_event, batch_size, input_latent_list_cycle, 
              audio_feat_queue, audio_out_queue, res_frame_queue,
              vae, unet, pe,timesteps): #vae, unet, pe,timesteps
    """推理函数,用于生成音频驱动的视频帧。视频帧只有脸部大小，后期需要替换到指定图片的脸部位置
    Args:
        render_event: 渲染控制事件
        batch_size: 批处理大小
        input_latent_list_cycle: 循环使用的潜在向量列表
        audio_feat_queue: 音频特征队列
        audio_out_queue: 音频输出队列
        res_frame_queue: 结果帧队列
        vae: VAE模型
        unet: UNet模型
        pe: 位置编码
        timesteps: 时间步长
    """
    length = len(input_latent_list_cycle)
    index = 0
    count = 0  # 计数器,用于计算FPS
    counttime = 0  # 累计时间,用于计算FPS
    logger.info('start inference')


    ########## 新加代码
    # 设置计算设备
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    whisper_dir = './models/whisper'

    audio_processor = AudioProcessor(feature_extractor_path = whisper_dir)  # 
    weight_dtype = unet.model.dtype  # 获取权重数据类型
    whisper = WhisperModel.from_pretrained(whisper_dir)  # 加载Whisper模型
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()  # 将模型移至设备并设置为评估模式
    whisper.requires_grad_(False)  # 禁用梯度计算
    ##########

    while render_event.is_set():
        starttime = time.perf_counter()
        try:
            # 获取音频特征
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)    # TODO: 这个audio_feat_queue，对应 museasr.py中 run_step() 内容。queue中的内容是已经处理好的音频特征？
        except queue.Empty:
            continue

        # 获取音频帧并检查是否全为静音
        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type == 0:  # type=0表示非静音
                is_all_silence = False

        if is_all_silence:
            # 如果全是静音,直接输出空帧
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # 处理非静音帧
            t = time.perf_counter()
            

            ############### >>>>>>>>>>>>>

            video_num = len(whisper_chunks)  # 获取视频帧数
            
            gen = datagen(whisper_chunks,  # 生成数据批次
                     input_latent_list_cycle,
                     batch_size)
            
            # 批量处理数据。*** 主要的视频帧就在这里完成 ***
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
                """
                pe是位置编码器（Positional Encoder）的缩写
                - 这个编码器用于为音频特征添加位置信息
                - 位置编码可以帮助模型理解音频特征在时间序列中的位置关系
                - 这对于生成与音频同步的嘴型动作非常重要
                """
                audio_feature_batch = pe(whisper_batch.to(device))  # 处理音频特征
                latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)  # 转换潜在向量

                pred_latents = unet.model(latent_batch,  # 使用UNet模型生成预测
                                        timesteps,
                                        encoder_hidden_states=audio_feature_batch).sample
                pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)  # 转换预测结果
                recon = vae.decode_latents(pred_latents)  # 解码潜在向量

                # 计算并输出FPS
                counttime += (time.perf_counter() - t)
                count += batch_size
                if count >= 100:
                    logger.info(f"------2 actual avg infer fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0

                # 输出生成的帧
                for i,res_frame in enumerate(recon):
                    res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                    index = index + 1

            ###############
            
    logger.info('musereal inference processor stop')

@torch.no_grad()
def inference_from_claude( # TODO: claude 生成，不一定对
        render_event,batch_size,input_latent_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue, vae, unet, pe,timesteps): #vae, unet, pe,timesteps
    """推理函数,用于生成音频驱动的视频帧。视频帧只有脸部大小，后期需要替换到指定图片的脸部位置
    Args:
        render_event: 渲染控制事件
        batch_size: 批处理大小
        input_latent_list_cycle: 循环使用的潜在向量列表
        audio_feat_queue: 音频特征队列  self.asr.feat_queue
        audio_out_queue: 音频输出队列   self.asr.output_queue
        res_frame_queue: 结果帧队列
        vae: VAE模型
        unet: UNet模型
        pe: 位置编码
        timesteps: 时间步长
    """
    length = len(input_latent_list_cycle)
    index = 0
    count = 0  # 计数器,用于计算FPS
    counttime = 0  # 累计时间,用于计算FPS
    logger.info('start inference')

    while render_event.is_set():
        starttime = time.perf_counter()
        try:
            # 获取音频特征
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        # 获取音频帧并检查是否全为静音
        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type == 0:  # type=0表示非静音
                is_all_silence = False

        if is_all_silence:
            # 如果全是静音,直接输出空帧
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # 处理非静音帧
            t = time.perf_counter()
            
            # 准备输入数据
            whisper_batch = np.stack(whisper_chunks)
            latent_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)
            
            # 处理音频特征 - 应用realtime_inference.py的方法
            audio_feature_batch = torch.from_numpy(whisper_batch)
            # 使用pe对音频特征进行位置编码，类似realtime_inference.py
            audio_feature_batch = pe(audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype))
            # 确保latent_batch的设备和数据类型正确
            latent_batch = latent_batch.to(device=unet.device, dtype=unet.model.dtype)

            # 使用UNet生成潜在向量
            pred_latents = unet.model(latent_batch, 
                                    timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
            
            # 转换pred_latents到VAE的数据类型，类似realtime_inference.py
            pred_latents = pred_latents.to(device=vae.vae.device, dtype=vae.vae.dtype)
            
            # 使用VAE解码生成图像
            recon = vae.decode_latents(pred_latents)

            # 计算并输出FPS
            counttime += (time.perf_counter() - t)
            count += batch_size
            if count >= 100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count = 0
                counttime = 0

            # 输出生成的帧
            for i,res_frame in enumerate(recon):
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
            
    logger.info('musereal inference processor stop')

@torch.no_grad()
def inference_2(render_event, batch_size, input_latent_list_cycle, 
                audio_feat_queue, audio_out_queue, res_frame_queue,
                vae, unet, pe, timesteps):
    """推理函数2，用于处理 run_step_2 产生的音频特征格式
    
    这个函数专门处理使用 MuseTalk 风格的音频特征提取方法产生的特征
    Args:
        render_event: 渲染控制事件
        batch_size: 批处理大小
        input_latent_list_cycle: 循环使用的潜在向量列表
        audio_feat_queue: 音频特征队列
        audio_out_queue: 音频输出队列
        res_frame_queue: 结果帧队列
        vae: VAE模型
        unet: UNet模型
        pe: 位置编码
        timesteps: 时间步长
    """
    length = len(input_latent_list_cycle)
    index = 0
    count = 0  # 计数器,用于计算FPS
    counttime = 0  # 累计时间,用于计算FPS
    logger.info('start inference_2')

    while render_event.is_set():
        starttime = time.perf_counter()
        try:
            # 获取音频特征 - 这里 whisper_chunks 已经是处理好的张量格式
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        # 获取音频帧并检查是否全为静音
        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type == 0:  # type=0表示非静音
                is_all_silence = False

        if is_all_silence:
            # 如果全是静音,直接输出空帧
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # 处理非静音帧
            t = time.perf_counter()
            
            # whisper_chunks 已经是 torch.Tensor 格式，形状为 [batch_size, seq_len, hidden_dim]
            # 直接使用，无需转换为 numpy
            audio_feature_batch = whisper_chunks.to(device=unet.device, dtype=unet.model.dtype)
            
            # 准备潜在向量批次
            latent_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)
            latent_batch = latent_batch.to(device=unet.device, dtype=unet.model.dtype)

            # 使用UNet生成潜在向量
            pred_latents = unet.model(latent_batch, 
                                    timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
            
            # 转换pred_latents到VAE的数据类型
            pred_latents = pred_latents.to(device=vae.vae.device, dtype=vae.vae.dtype)
            
            # 使用VAE解码生成图像
            recon = vae.decode_latents(pred_latents)

            # 计算并输出FPS
            counttime += (time.perf_counter() - t)
            count += batch_size
            if count >= 100:
                logger.info(f"------inference_2 actual avg infer fps:{count/counttime:.4f}")
                count = 0
                counttime = 0

            # 输出生成的帧
            for i,res_frame in enumerate(recon):
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
            
    logger.info('musereal inference_2 processor stop')

class MuseReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps # 20 ms per frame

        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size*2)

        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model # from load_model() above
        self.frame_list_cycle,self.mask_list_cycle,self.coord_list_cycle,self.mask_coords_list_cycle, self.input_latent_list_cycle = avatar
        #self.__loadavatar()

        # 延迟导入以避免循环导入
        from museasr import MuseASR
        self.asr = MuseASR(opt,self,self.audio_processor)
        self.asr.warm_up()
        
        self.render_event = mp.Event()

    def __del__(self):
        logger.info(f'musereal({self.sessionid}) delete')
    

    def __mirror_index(self, index):
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1  

    def __warm_up(self): 
        self.asr.run_step()
        whisper_chunks = self.asr.get_next_feat()
        whisper_batch = np.stack(whisper_chunks)
        latent_batch = []
        for i in range(self.batch_size):
            idx = self.__mirror_index(self.idx+i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)
        logger.info('infer=======')
        # for i, (whisper_batch,latent_batch) in enumerate(gen):
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                        dtype=self.unet.model.dtype)
        audio_feature_batch = self.pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

        pred_latents = self.unet.model(latent_batch, 
                                    self.timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
        recon = self.vae.decode_latents(pred_latents)
      

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        '''
        处理生成的视频帧并进行状态管理的主要函数。（只合并视频帧，不做推理）
        
        参数:
            quit_event: 退出事件,用于控制线程退出
            loop: 事件循环
            audio_track: 音频轨道
            video_track: 视频轨道
            
        主要功能:
        1. 从结果队列中获取生成的视频帧和音频帧
        2. 根据音频帧判断当前是说话状态还是静音状态
        3. 在说话和静音状态切换时进行平滑过渡
        4. 处理静音状态下的自定义表情帧
        5. 处理说话状态下的生成人脸帧
        6. 将处理后的帧发送到视频轨道
        '''
        # 新增状态跟踪变量
        self.last_speaking = False
        self.transition_start = time.time()
        self.transition_duration = 0.1  # 过渡时间
        self.last_silent_frame = None  # 静音帧缓存
        self.last_speaking_frame = None  # 说话帧缓存
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 检测状态变化
            current_speaking = not (audio_frames[0][1]!=0 and audio_frames[1][1]!=0)
            if current_speaking != self.last_speaking:
                logger.info(f"状态切换：{'说话' if self.last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                self.transition_start = time.time()
            self.last_speaking = current_speaking
            
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: 
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None:
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                # 说话→静音过渡
                if time.time() - self.transition_start < self.transition_duration and self.last_speaking_frame is not None:
                    alpha = min(1.0, (time.time() - self.transition_start) / self.transition_duration)
                    combine_frame = cv2.addWeighted(self.last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                else:
                    combine_frame = target_frame
                # 缓存静音帧
                self.last_silent_frame = combine_frame.copy()
            else:
                # 设置说话状态为真
                self.speaking = True
                # 获取当前帧的人脸边界框坐标
                bbox = self.coord_list_cycle[idx]
                # 深拷贝原始帧,避免修改原始数据
                ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
                # 解包边界框坐标
                x1, y1, x2, y2 = bbox
                try:
                    # 将生成的人脸帧调整到边界框大小
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except Exception as e:
                    logger.warning(f"resize error: {e}")
                    continue
                # 获取当前帧的遮罩和遮罩坐标
                mask = self.mask_list_cycle[idx]
                mask_crop_box = self.mask_coords_list_cycle[idx]

                # 静音→说话过渡
                # 将生成的人脸与原始帧进行融合
                current_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)
                if time.time() - self.transition_start < self.transition_duration and self.last_silent_frame is not None:
                    # 计算过渡透明度
                    alpha = min(1.0, (time.time() - self.transition_start) / self.transition_duration)
                    # 使用透明度混合静音帧和说话帧
                    combine_frame = cv2.addWeighted(self.last_silent_frame, 1-alpha, current_frame, alpha, 0)
                else:
                    combine_frame = current_frame
                # 缓存当前说话帧用于下次过渡
                self.last_speaking_frame = combine_frame.copy()

            # 准备输出视频帧
            image = combine_frame
            # 将numpy数组转换为VideoFrame格式
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            # 异步发送视频帧到输出队列
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            # 记录视频数据
            self.record_video_data(image)

            # 处理音频帧
            for audio_frame in audio_frames:
                # 解包音频数据、类型和事件点
                frame,type,eventpoint = audio_frame
                # 将浮点音频数据转换为16位整数格式
                frame = (frame * 32767).astype(np.int16)
                # 创建新的音频帧
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                # 更新音频数据
                new_frame.planes[0].update(frame.tobytes())
                # 设置采样率
                new_frame.sample_rate=16000
                # 异步发送音频帧到输出队列
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                # 记录音频数据
                self.record_audio_data(frame)
        logger.info('musereal process_frames thread stop') 
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        # 该函数在 webrtc.py 中 被 container.render(...) 调用
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        self.render_event.set() #start infer process render
        Thread(target=inference, args=(
            self.render_event,
            self.batch_size,
            self.input_latent_list_cycle,
            self.asr.feat_queue,    # TODO: 在 baseasr.py 中，BaseASR 下的 feat_queue
            self.asr.output_queue,
            self.res_frame_queue,
            self.vae, 
            self.unet, 
            self.pe,
            self.timesteps)).start() #mp.Process
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): #todo
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()     # 提取音频特征
            #self.test_step(loop,audio_track,video_track)
            # totaltime += (time.perf_counter() - t)
            # count += self.opt.batch_size
            # if count>=100:
            #     print(f"------actual avg infer fps:{count/totaltime:.4f}")
            #     count=0
            #     totaltime=0
            if video_track._queue.qsize()>=1.5*self.opt.batch_size:
                logger.debug('sleep qsize=%d',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
            # if video_track._queue.qsize()>=5:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        self.render_event.clear() #end infer process render
        logger.info('musereal thread stop')

    def render_2(self,quit_event,loop=None,audio_track=None,video_track=None):
        """使用 run_step_2 和 inference_2 的渲染方法
        
        这个方法使用与 MuseTalk 一致的音频特征提取方法
        """
        # 该函数在 webrtc.py 中 被 container.render(...) 调用
        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        self.render_event.set() #start infer process render
        Thread(target=inference_2, args=(
            self.render_event,
            self.batch_size,
            self.input_latent_list_cycle,
            self.asr.feat_queue,
            self.asr.output_queue,
            self.res_frame_queue,
            self.vae, 
            self.unet, 
            self.pe,
            self.timesteps)).start() #mp.Process
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        
        while not quit_event.is_set():
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step_2()     # 使用新的音频特征提取方法
            if video_track._queue.qsize()>=1.5*self.opt.batch_size:
                logger.debug('sleep qsize=%d',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
                
        self.render_event.clear() #end infer process render
        logger.info('musereal render_2 thread stop')
            
