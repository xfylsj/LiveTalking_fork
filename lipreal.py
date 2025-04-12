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
import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp


from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from wav2lip.models import Wav2Lip
from basereal import BaseReal

#from imgcache import ImgCache

from tqdm import tqdm
from logger import logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path) #,weights_only=True
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	logger.info("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    # 使用glob模块获取全身图像目录下所有jpg/jpeg/png格式的图片路径列表
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # 按文件名中的数字排序图片列表，确保按正确顺序加载
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # 读取所有全身图像并存入frame_list_cycle列表
    frame_list_cycle = read_imgs(input_img_list)
    #self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    # 使用glob模块获取面部图像目录下所有jpg/jpeg/png格式的图片路径列表
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # 按文件名中的数字排序面部图像列表，确保按正确顺序加载
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # 读取所有面部图像并存入face_list_cycle列表
    face_list_cycle = read_imgs(input_face_list)
    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):  # 循环索引
    # size: 图像序列的总长度
    # index: 当前需要计算的索引值
    
    turn = index // size    # 计算已经完整循环了多少轮
    res = index % size      # 计算在当前轮次中的位置
    
    if turn % 2 == 0:       # 如果是偶数轮次（第0轮、第2轮、第4轮...）
        return res          # 正向播放，直接返回索引
    else:                   # 如果是奇数轮次（第1轮、第3轮、第5轮...）
        return size - res - 1  # 反向播放，返回反转的索引

def inference(quit_event,batch_size,face_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue,model):
    # 参数说明:
    # quit_event: 退出事件信号，用于停止推理线程
    # batch_size: 批处理大小，一次处理多少帧
    # face_list_cycle: 面部图像列表，用作生成唇形的基础
    # audio_feat_queue: 音频特征队列，包含梅尔频谱图等特征
    # audio_out_queue: 音频输出队列，包含原始音频帧
    # res_frame_queue: 结果帧队列，用于存放生成的唇形图像
    # model: Wav2Lip模型实例
    
    # 注释掉的代码是直接加载模型的替代方案
    #model = load_model("./models/wav2lip.pth")
    # 注释掉的代码是直接加载面部图像的替代方案
    # input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # face_list_cycle = read_imgs(input_face_list)
    
    # 注释掉的代码是加载预计算潜在特征的替代方案
    #input_latent_list_cycle = torch.load(latents_out_path)
    
    # 获取面部图像列表的长度，用于循环索引
    length = len(face_list_cycle)
    # 初始化索引，用于跟踪当前处理的帧位置
    index = 0
    # 初始化计数器，用于性能统计
    count=0
    counttime=0
    # 记录推理开始的日志
    logger.info('start inference')
    
    # 主循环，直到收到退出信号
    while not quit_event.is_set():
        # 记录当前批次开始时间
        starttime=time.perf_counter()
        # 初始化梅尔频谱批次
        mel_batch = []
        try:
            # 从音频特征队列获取一批梅尔频谱，超时1秒
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            # 队列为空时继续循环
            continue
            
        # 假设所有帧都是静音，后面会检查
        is_all_silence=True
        # 初始化音频帧列表
        audio_frames = []
        # 获取batch_size*2个音频帧（每个视频帧对应2个音频帧）
        for _ in range(batch_size*2):
            # 从音频输出队列获取音频帧、类型和事件点
            frame,type,eventpoint = audio_out_queue.get()
            # 将音频帧添加到列表
            audio_frames.append((frame,type,eventpoint))
            # 如果类型为0，表示有声音，不是静音
            if type==0:
                is_all_silence=False

        # 如果全是静音帧，不需要进行唇形合成
        if is_all_silence:
            # 对每个批次中的帧进行处理
            for i in range(batch_size):
                # 将空帧、镜像索引和对应的音频帧放入结果队列
                # None表示不需要合成唇形，直接使用原始图像
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                # 索引递增
                index = index + 1
        else:
            # 有声音的帧，需要进行唇形合成
            # 注释掉的调试输出
            # print('infer=======')
            # 记录推理开始时间
            t=time.perf_counter()
            # 初始化图像批次
            img_batch = []
            # 为批次中的每一帧准备面部图像
            for i in range(batch_size):
                # 计算镜像索引，实现循环播放时的正放和倒放效果
                idx = __mirror_index(length,index+i)
                # 获取对应索引的面部图像
                face = face_list_cycle[idx]
                # 将面部图像添加到批次
                img_batch.append(face)
            # 将图像批次和梅尔频谱批次转换为NumPy数组
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            # 创建图像的掩码版本，右半部分设为0
            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0

            # 将掩码图像和原始图像在水平方向连接，并归一化到0-1范围
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            # 重塑梅尔频谱批次的维度，适应模型输入要求
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            # 将NumPy数组转换为PyTorch张量，并调整维度顺序
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            # 使用模型进行推理，生成唇形图像
            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            # 将预测结果转换回NumPy数组，调整维度顺序，并缩放到0-255范围
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            # 累加推理时间和帧数，用于计算平均FPS
            counttime += (time.perf_counter() - t)
            count += batch_size
            # 注释掉的总帧数计数
            #_totalframe += 1
            # 每处理100帧输出一次平均FPS
            if count>=100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count=0
                counttime=0
            # 处理批次中的每一帧结果
            for i,res_frame in enumerate(pred):
                # 注释掉的直接推送媒体的替代方案
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                # 将生成的唇形图像、镜像索引和对应的音频帧放入结果队列
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                # 索引递增
                index = index + 1
            # 注释掉的批次总时间输出
            #print('total batch time:',time.perf_counter()-starttime)            
    # 推理线程结束时记录日志
    logger.info('lipreal inference processor stop')

class LipReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps # 20 ms per frame
        
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size*2)  #mp.Queue
        #self.__loadavatar()
        self.model = model
        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = LipASR(opt,self)
        self.asr.warm_up()
        
        self.render_event = mp.Event()
    
    def __del__(self):
        logger.info(f'lipreal({self.sessionid}) delete')

    # 处理帧的函数
    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        """
        处理帧的主函数。
        
        参数:
            quit_event: 退出事件信号
            loop: 事件循环
            audio_track: 音频轨道
            video_track: 视频轨道
            
        功能:
            1. 启动推理处理器线程,生成唇形同步的面部图像
            2. 持续从结果队列获取处理好的帧
            3. 根据音频类型(有声/静音)选择:
               - 静音时使用自定义视频或默认全身图像
               - 有声时将生成的面部图像合成到全身图像中
            4. 将合成的帧推送到音视频轨道进行播放
        """

        # 持续处理帧直到收到退出信号
        while not quit_event.is_set():
            try:
                # 从结果队列获取一个处理好的帧数据，包含面部图像、索引和对应的音频帧
                # 如果1秒内没有数据则超时继续循环
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                # 队列为空时继续循环
                continue
                
            # 判断是否为静音帧，audio_frames[x][1]中的1表示帧类型，0为有声音，非0为静音
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: # 全为静音数据，只需要取全身图像
                # 设置说话状态为否
                self.speaking = False
                # 获取音频类型（可能是不同的静音状态，如思考、等待等）
                audiotype = audio_frames[0][1]
                # 检查是否有与该音频类型对应的自定义视频
                if self.custom_index.get(audiotype) is not None: # 有自定义视频
                    # 计算镜像索引，用于循环播放视频时的前进后退效果
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    # 获取对应的自定义视频帧
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    # 自定义视频索引递增，准备下一帧
                    self.custom_index[audiotype] += 1
                    # 注释掉的代码用于处理非循环播放的情况
                    # if not self.custom_opt[audiotype].loop and self.custom_index[audiotype]>=len(self.custom_img_cycle[audiotype]):
                    #     self.curr_state = 1  # 当前视频不循环播放，切换到静音状态
                else:
                    # 没有自定义视频时，使用默认的全身图像
                    combine_frame = self.frame_list_cycle[idx]
                    # 注释掉的代码是使用图像缓存的替代方案
                    # combine_frame = self.imagecache.get_img(idx)
            else:
                # 有声音的帧，需要进行唇形合成
                self.speaking = True
                # 获取当前索引对应的面部区域坐标
                bbox = self.coord_list_cycle[idx]
                # 深拷贝全身图像，避免修改原始数据
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                # 注释掉的代码是使用图像缓存的替代方案
                # combine_frame = copy.deepcopy(self.imagecache.get_img(idx))
                # 解析面部区域坐标
                y1, y2, x1, x2 = bbox
                try:
                    # 将生成的面部图像调整为面部区域的大小
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    # 调整大小失败时跳过当前帧
                    continue
                # 注释掉的代码是另一种合成方法
                # combine_frame = get_image(ori_frame,res_frame,bbox)
                # 注释掉的代码是用于性能测量
                # t=time.perf_counter()
                # 将生成的面部图像放入全身图像的对应位置
                combine_frame[y1:y2, x1:x2] = res_frame
                # 注释掉的代码是用于性能测量
                # print('blending time:',time.perf_counter()-t)
            # 最终合成的图像｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀｀

            # 将合成后的图像赋值给image变量
            image = combine_frame #(outputs['image'] * 255).astype(np.uint8)
            # 将numpy数组转换为VideoFrame对象,指定格式为bgr24
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            # 将视频帧异步放入视频轨道的队列中
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            # 记录视频数据用于后续处理
            self.record_video_data(image)

            # 遍历音频帧列表
            for audio_frame in audio_frames:
                # 解包音频帧数据,获取帧数据、类型和事件点
                frame,type,eventpoint = audio_frame
                # 将音频数据转换为16位整型,范围[-32768,32767]
                frame = (frame * 32767).astype(np.int16)
                # 创建新的音频帧,设置格式为s16,单声道,采样数为帧长度
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                # 更新音频帧的数据平面
                new_frame.planes[0].update(frame.tobytes())
                # 设置音频采样率为16kHz
                new_frame.sample_rate=16000
                # if audio_track._queue.qsize()>10:
                #     time.sleep(0.1)
                # 将音频帧异步放入音频轨道的队列中
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                # 记录音频数据用于后续处理
                self.record_audio_data(frame)
                #self.notify(eventpoint)
        logger.info('lipreal process_frames thread stop') 
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        """
        render video and audio
        """
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        Thread(target=inference, args=(quit_event,self.batch_size,self.face_list_cycle,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.model,)).start()  #mp.Process

        #self.render_event.set() #start infer process render
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): 
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step() # 运行ASR步骤

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)

            # 如果视频轨道队列中的帧数大于等于5,则打印队列大小并等待一段时间
            if video_track._queue.qsize()>=5:
                logger.debug('sleep qsize=%d',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        #self.render_event.clear() #end infer process render
        logger.info('lipreal thread stop')
                        