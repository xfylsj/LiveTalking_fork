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
from __future__ import annotations
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts
import fractions


from av import AudioFrame
from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basereal import BaseReal

from logger import logger
class State(Enum):
    RUNNING=0
    PAUSE=1

audio_frame_queue = Queue()  # 音频帧队列

class BaseTTS:

    def __init__(self):

        self.fps = 50 # 每帧20毫秒
        self.sample_rate = 16000  # 采样率16kHz
        self.chunk = self.sample_rate // self.fps # 每块320个样本(20ms * 16000 / 1000)
        self.input_stream = BytesIO()   # 16khz 20ms pcm音频流缓冲

        self.msgqueue = Queue()  # 消息队列
        self.state = State.RUNNING  # 初始状态为运行

    # 清空对话队列并暂停TTS
    def flush_talk(self):
        self.msgqueue.queue.clear()  # 清空消息队列
        self.state = State.PAUSE  # 设置状态为暂停

    # 添加文本消息到队列
    def put_msg_txt(self,msg:str,eventpoint=None): 
        if len(msg)>0:  # 确保消息非空
            self.msgqueue.put((msg,eventpoint))  # 将消息和事件点添加到队列

    # 启动渲染线程
    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))  # 创建TTS处理线程
        process_thread.start()  # 启动线程
    
    # TTS处理主循环
    def process_tts(self, quit_event):        
        while not quit_event.is_set():  # 当退出事件未设置时循环
            try:
                msg = self.msgqueue.get(block=True, timeout=1)  # 从队列获取消息，阻塞1秒
                self.state=State.RUNNING  # 设置状态为运行
            except queue.Empty:  # 队列为空时继续循环
                continue
            self.txt_to_audio(msg)  # 处理消息，转换为音频
        logger.info('ttsreal thread stop')  # 记录线程停止信息
    
    # 文本转语音抽象方法（由子类实现）
    def txt_to_audio(self,msg):
        pass
    


###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self, msg):
        self.stream_tts(
            self.cosy_voice(
                msg,
                "~/zero_shot_prompt.wav",  
                "这个星期我简直忙坏了，他对盲锣先生说。我要上观察课",
                "zh", #en 
                "http://10.60.32.123:50000",
            ),
            msg,
        )
   

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]

            # 收到的是一个持续输出的响应流
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=8820): # 882 22.05K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

    def stream_tts(self, audio_stream, msg):
        text = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text}
                        first = False
                    
                    self.audio_frame_queue.put((stream[idx:idx+self.chunk],eventpoint))
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text}
        self.audio_frame_queue.put(np.zeros(self.chunk,np.float32),eventpoint) 


if __name__ == "__main__":
    cosyvoice = CosyVoiceTTS()
    cosyvoice.render(Event())
  


