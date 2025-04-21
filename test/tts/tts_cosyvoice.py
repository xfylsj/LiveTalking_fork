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
import fractions
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import pyaudio
import asyncio

from aiortc import MediaStreamTrack



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

import logging
import os

# 获取当前程序文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(current_dir, 'test.log')

# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 文件处理器
fhandler = logging.FileHandler(log_file)
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

# 添加控制台处理器
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)  # 设置与文件处理器相同的级别
handler.setFormatter(formatter)  # 使用相同的格式
logger.addHandler(handler)

class State(Enum):
    RUNNING=0
    PAUSE=1



class BaseTTS:

    def __init__(self):

        self.fps = 50 # 每帧20毫秒
        self.sample_rate = 16000  # 采样率16kHz
        self.chunk = self.sample_rate // self.fps # 每块320个样本(20ms * 16000 / 1000)
        self.input_stream = BytesIO()   # 16khz 20ms pcm音频流缓冲

        self.msgqueue = Queue()  # 消息队列
        self.audio_frame_queue = Queue()  # 音频帧队列
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
        logger.info(f'render quit_event: {quit_event}')
        process_thread = Thread(target=self.process_tts, args=(quit_event,))  # 创建TTS处理线程
        process_thread.start()  # 启动线程
    
    # TTS处理主循环
    def process_tts(self, quit_event):     
        logger.info('ttsreal thread start')  # 记录线程停止信息

        while not quit_event.is_set():  # 当退出事件未设置时循环
            try:
                msg = self.msgqueue.get(block=True, timeout=1)  # 从队列获取消息，阻塞1秒
                self.state=State.RUNNING  # 设置状态为运行
            except queue.Empty:  # 队列为空时继续循环
                # logger.info('Empty msg queue')
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
            # self.cosy_voice(
            self.cosy_voice_local(
                msg,
                "test/tts/zero_shot_prompt.wav",  
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

    def cosy_voice_local(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]

            # 收到的是一个持续输出的响应流
            res = requests.request("GET", f"http://localhost:18080/audio", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=16000): # 882 22.05K*20ms*2
                if first:
                    # 跳过WAV文件头(44字节)
                    chunk = chunk[44:]
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
        buffer = np.array([], dtype=np.int16)
        
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                # 直接使用 int16
                stream = np.frombuffer(chunk, dtype=np.int16)
                
                # 将新数据添加到缓冲区
                buffer = np.concatenate([buffer, stream])
                
                # 处理完整的块
                while len(buffer) >= self.chunk:
                    current_chunk = buffer[:self.chunk]
                    buffer = buffer[self.chunk:]
                    
                    eventpoint = None
                    if first:
                        eventpoint = {'status':'start','text':text}
                        first = False
                    
                    self.audio_frame_queue.put((current_chunk, eventpoint))
                
        # 处理剩余的数据
        if len(buffer) > 0:
            self.audio_frame_queue.put((buffer, None))
        
        eventpoint = {'status':'end','text':text}
        self.audio_frame_queue.put((np.zeros(self.chunk, dtype=np.int16), eventpoint))

class AudioPlayer:
    def __init__(self, audio_frame_queue, sample_rate=16000):  # 使用正确的采样率
        self.audio_frame_queue = audio_frame_queue
        self.sample_rate = sample_rate
        self.stream = None
        self.p = None
        self.running = False
        self.chunk_size = 1600  # 调整为100ms的数据量 (16000 * 0.1)
        self.volume = 5  # 音量放大倍数，因为原始数据幅度较小

    async def start(self):
        """异步启动音频播放"""
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        logger.info(f"打开音频流: 采样率={self.sample_rate}, 块大小={self.chunk_size}")
        self.running = True
        await self.play_audio()

    async def stop(self):
        """停止音频播放"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        self.stream = None
        self.p = None

    async def play_audio(self):
        """从队列中读取并播放音频帧"""
        logger.info("等待音频数据...")
        
        while self.running:
            try:
                frame, event = self.audio_frame_queue.get_nowait()
                
                if event and 'status' in event:
                    if event['status'] == 'start':
                        logger.info(f"开始播放: {event.get('text', '')}")
                    elif event['status'] == 'end':
                        logger.info(f"播放完成: {event.get('text', '')}")
                        continue
                        
                if len(frame) > 0:
                    # 确保数据类型是 int16
                    frame = frame.astype(np.int16)
                    
                    self.stream.write(frame.tobytes())

                    
            except queue.Empty:
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.error(f"播放音频时发生错误: {str(e)}")
                await self.stop()
                await asyncio.sleep(1)
                try:
                    if self.running:
                        self.p = pyaudio.PyAudio()
                        self.stream = self.p.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            output=True,
                            frames_per_buffer=self.chunk_size
                        )
                except Exception as e:
                    logger.error(f"重新初始化音频设备失败: {str(e)}")
                    await asyncio.sleep(1)

class AudioFrameTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, audio_frame_queue: Queue):
        super().__init__()
        self.audio_frame_queue = audio_frame_queue
        self.sample_rate = 16000  # 采样率
        self.frame_duration = 0.02  # 20ms 每帧
        self.frame_no = 0
    
    async def recv(self):
        try:
            # 从队列中获取音频帧数据
            audio_data, event = self.audio_frame_queue.get_nowait()
            
            # 检查是否结束
            if event and 'status' in event and event['status'] == 'end':
                return None
                
            # 重新排列数据形状为 (channels, samples)
            audio_data = audio_data.reshape(1, -1)  # 1个通道，samples个样本
            
            # 创建音频帧
            frame = AudioFrame.from_ndarray(
                audio_data,
                format='s16', 
                layout='mono'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self.frame_no
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            
            self.frame_no += 1
            
            # 控制帧率
            await asyncio.sleep(self.frame_duration)
            
            return frame
            
        except queue.Empty:
            # 队列为空时等待
            await asyncio.sleep(0.1)
            return None

async def main(audio_frame_queue):
    # 创建音频轨道
    audio_track = AudioFrameTrack(audio_frame_queue)
    
    try:
        logger.info("等待播放音频...")
        while True:
            frame = await audio_track.recv()
            if frame is not None:
                logger.info(f"播放音频帧 {frame.pts}, 时间戳: {frame.pts * frame.time_base}")
                pass
            
    except KeyboardInterrupt:
        logger.info("\n停止播放")
    finally:
        audio_track.stop()

def input_worker(tts, quit_event):
    """读取用户控制台输入的线程函数"""
    logger.info("开始接收用户输入，请输入文本（输入 'q' 退出）：")
    while not quit_event.is_set():
        try:
            text = input().strip()
            if text.lower() == 'q':
                quit_event.set()
                break
            if text:
                logger.info(f"收到用户输入：{text}")
                tts.put_msg_txt(text)
        except EOFError:
            break
        except KeyboardInterrupt:
            quit_event.set()
            break
    logger.info("用户输入线程结束")

if __name__ == "__main__":
    logger.info("------ new -------")
    
    cosyvoice = CosyVoiceTTS()
    quit_event = Event()
    
    # 启动TTS渲染线程
    cosyvoice.render(quit_event)
    
    # 创建音频播放器
    audio_player = AudioPlayer(cosyvoice.audio_frame_queue)
    
    # 创建并启动用户输入线程
    input_thread = Thread(target=input_worker, args=(cosyvoice, quit_event))
    input_thread.daemon = True
    input_thread.start()
    
    try:
        logger.info("准备就绪，请输入要转换的文本（输入 'q' 退出）：")
        # 运行音频播放循环
        asyncio.run(audio_player.start())
    except KeyboardInterrupt:
        logger.info("\n收到退出信号")
    finally:
        quit_event.set()
        input_thread.join(timeout=1)
        logger.info("程序结束")


