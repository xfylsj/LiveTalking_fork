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

class BaseTTS:
    def __init__(self, opt, parent:BaseReal):
        self.opt=opt  # 配置选项
        self.parent = parent  # 父对象引用

        self.fps = opt.fps # 每帧20毫秒
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
    def process_tts(self,quit_event):        
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
# Edge TTS实现类 - 使用微软Edge TTS服务
class EdgeTTS(BaseTTS):
    # 文本转语音实现
    def txt_to_audio(self,msg):
        voicename = "zh-CN-YunxiaNeural"  # 中文语音模型
        text,textevent = msg  # 解包消息，获取文本和事件
        t = time.time()  # 记录开始时间
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))  # 异步调用TTS服务，得到音频流 BytesIO,写入input_stream
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')  # 记录TTS处理时间
        if self.input_stream.getbuffer().nbytes<=0:  # 检查是否获取到音频数据
            logger.error('edgetts err!!!!!')  # 记录错误
            return
        
        # 处理音频流
        self.input_stream.seek(0)  # 将缓冲区指针移到开始位置
        stream = self.__create_bytes_stream(self.input_stream)  # 转换为音频流
        streamlen = stream.shape[0]  # 获取音频流长度
        idx=0  # 初始化索引
        while streamlen >= self.chunk and self.state==State.RUNNING:  # 当有足够数据且状态为运行时
            eventpoint=None  # 初始化事件点
            streamlen -= self.chunk  # 更新剩余长度
            if idx==0:  # 第一个音频块
                eventpoint={'status':'start','text':text,'msgenvent':textevent}  # 设置开始事件
            elif streamlen<self.chunk:  # 最后一个完整音频块
                eventpoint={'status':'end','text':text,'msgenvent':textevent}  # 设置结束事件
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)  # 发送音频帧
            idx += self.chunk  # 更新索引
        self.input_stream.seek(0)  # 重置缓冲区指针
        self.input_stream.truncate()  # 清空缓冲区

    # 创建音频流 - 将二进制数据转换为音频数组
    def __create_bytes_stream(self,byte_stream):
        stream, sample_rate = sf.read(byte_stream)  # 读取音频数据和采样率
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')  # 记录音频信息
        stream = stream.astype(np.float32)  # 转换为float32类型

        # 处理多声道音频
        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')  # 记录警告
            stream = stream[:, 0]  # 只保留第一个声道
    
        # 重采样到目标采样率
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')  # 记录警告
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)  # 重采样

        return stream  # 返回处理后的音频流
    
    # Edge TTS主处理函数 - 异步获取TTS音频数据
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)  # 创建通信对象

            first = True  # 标记第一个数据块
            async for chunk in communicate.stream():  # 异步迭代获取数据块
                if first:
                    first = False  # 重置标记
                if chunk["type"] == "audio" and self.state==State.RUNNING:  # 如果是音频数据且状态为运行
                    self.input_stream.write(chunk["data"])  # 写入音频数据
                elif chunk["type"] == "WordBoundary":  # 词边界信息
                    pass  # 暂不处理
        except Exception as e:
            logger.exception('edgetts')  # 记录异常

###########################################################################################
# Fish TTS实现类 - 使用Fish Speech服务
class FishTTS(BaseTTS):
    # 文本转语音实现
    def txt_to_audio(self,msg): 
        text,textevent = msg  # 解包消息
        self.stream_tts(  # 调用流式TTS处理
            self.fish_speech(  # 调用Fish Speech API
                text,  # 文本内容
                self.opt.REF_FILE,  # 参考音频文件
                self.opt.REF_TEXT,  # 参考文本
                "zh",  # 语言设置
                self.opt.TTS_SERVER,  # TTS服务器地址
            ),
            msg  # 原始消息
        )

    # Fish Speech API调用 - 返回音频数据迭代器
    def fish_speech(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()  # 记录开始时间
        req={  # 构建请求参数
            'text': text,  # 文本内容
            'reference_id': reffile,  # 参考音频ID
            'format': 'wav',  # 输出格式
            'streaming': True,  # 启用流式传输
            'use_memory_cache': 'on'  # 使用内存缓存
        }
        try:
            # 发送POST请求
            res = requests.post(
                f"{server_url}/v1/tts",  # API端点
                json=req,  # JSON请求体
                stream=True,  # 启用流式响应
                headers={
                    "content-type": "application/json",  # 内容类型
                },
            )
            end = time.perf_counter()  # 记录请求完成时间
            logger.info(f"fish_speech Time to make POST: {end-start}s")  # 记录请求耗时

            if res.status_code != 200:  # 检查响应状态
                logger.error("Error:%s", res.text)  # 记录错误
                return
                
            first = True  # 标记第一个数据块
        
            # 处理流式响应
            for chunk in res.iter_content(chunk_size=17640):  # 每块约20ms的44.1kHz音频
                if first:
                    end = time.perf_counter()  # 记录首个数据块时间
                    logger.info(f"fish_speech Time to first chunk: {end-start}s")  # 记录首块延迟
                    first = False  # 重置标记
                if chunk and self.state==State.RUNNING:  # 如果有数据且状态为运行
                    yield chunk  # 返回数据块
        except Exception as e:
            logger.exception('fishtts')  # 记录异常

    # 流式TTS处理 - 处理音频数据流并发送音频帧
    def stream_tts(self, audio_stream, msg):
        text, textevent = msg  # 解包消息
        first = True  # 标记第一个音频帧
        for chunk in audio_stream:  # 迭代音频数据块
            if chunk is not None and len(chunk)>0:  # 确保数据块有效
                # 将二进制数据转换为float32音频数组并重采样
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767  # 转换为[-1,1]范围
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)  # 重采样到16kHz
                
                streamlen = stream.shape[0]  # 获取音频流长度
                idx=0  # 初始化索引
                while streamlen >= self.chunk:  # 当有足够数据时
                    eventpoint=None  # 初始化事件点
                    if first:  # 第一个音频帧
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}  # 设置开始事件
                        first = False  # 重置标记
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)  # 发送音频帧
                    streamlen -= self.chunk  # 更新剩余长度
                    idx += self.chunk  # 更新索引
        # 发送结束事件和空音频帧
        eventpoint={'status':'end','text':text,'msgenvent':textevent}  # 设置结束事件
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  # 发送空音频帧

###########################################################################################
# Voits TTS实现类 - 使用GPT-SoVITS服务
class VoitsTTS(BaseTTS):
    # 文本转语音实现
    def txt_to_audio(self,msg): 
        text,textevent = msg  # 解包消息
        self.stream_tts(  # 调用流式TTS处理
            self.gpt_sovits(  # 调用GPT-SoVITS API
                text,  # 文本内容
                self.opt.REF_FILE,  # 参考音频文件
                self.opt.REF_TEXT,  # 参考文本
                "zh",  # 语言设置
                self.opt.TTS_SERVER,  # TTS服务器地址
            ),
            msg  # 原始消息
        )

    # GPT-SoVITS API调用 - 返回音频数据迭代器
    def gpt_sovits(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()  # 记录开始时间
        req={  # 构建请求参数
            'text': text,  # 文本内容
            'text_lang': language,  # 文本语言
            'ref_audio_path': reffile,  # 参考音频路径
            'prompt_text': reftext,  # 提示文本
            'prompt_lang': language,  # 提示语言
            'media_type': 'ogg',  # 媒体类型
            'streaming_mode': True  # 启用流式模式
        }
        # 注释掉的旧参数格式
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                #stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                #stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream=BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)

###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
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

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
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
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
_PROTOCOL = "https://"
_HOST = "tts.cloud.tencent.com"
_PATH = "/stream"
_ACTION = "TextToStreamAudio"

class TencentTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(opt.REF_FILE)
        self.codec = "pcm"
        self.sample_rate = 16000
        self.volume = 0
        self.speed = 0
    
    def __gen_signature(self, params):
        sort_dict = sorted(params.keys())
        sign_str = "POST" + _HOST + _PATH + "?"
        for key in sort_dict:
            sign_str = sign_str + key + "=" + str(params[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'),
                           sign_str.encode('utf-8'), hashlib.sha1).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode('utf-8')
        return s

    def __gen_params(self, session_id, text):
        params = dict()
        params['Action'] = _ACTION
        params['AppId'] = int(self.appid)
        params['SecretId'] = self.secret_id
        params['ModelType'] = 1
        params['VoiceType'] = self.voice_type
        params['Codec'] = self.codec
        params['SampleRate'] = self.sample_rate
        params['Speed'] = self.speed
        params['Volume'] = self.volume
        params['SessionId'] = session_id
        params['Text'] = text

        timestamp = int(time.time())
        params['Timestamp'] = timestamp
        params['Expired'] = timestamp + 24 * 60 * 60
        return params

    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.tencent_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def tencent_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        session_id = str(uuid.uuid1())
        params = self.__gen_params(session_id, text)
        signature = self.__gen_signature(params)
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        url = _PROTOCOL + _HOST + _PATH
        try:
            res = requests.post(url, headers=headers,
                          data=json.dumps(params), stream=True)
            
            end = time.perf_counter()
            logger.info(f"tencent Time to make POST: {end-start}s")
                
            first = True
        
            for chunk in res.iter_content(chunk_size=6400): # 640 16K*20ms*2
                #logger.info('chunk len:%d',len(chunk))
                if first:
                    try:
                        rsp = json.loads(chunk)
                        #response["Code"] = rsp["Response"]["Error"]["Code"]
                        #response["Message"] = rsp["Response"]["Error"]["Message"]
                        logger.error("tencent tts:%s",rsp["Response"]["Error"]["Message"])
                        return
                    except:
                        end = time.perf_counter()
                        logger.info(f"tencent Time to first chunk: {end-start}s")
                        first = False                    
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('tencent')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        last_stream = np.array([],dtype=np.float32)
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream,stream))
                #stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:] #get the remain stream
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg):
        text,textevent = msg  
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            ),
            msg
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)


