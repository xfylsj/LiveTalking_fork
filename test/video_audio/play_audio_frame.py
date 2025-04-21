import fractions
import json
import os
import asyncio
import glob
import numpy as np
from av import AudioFrame
from aiortc import MediaStreamTrack
from scipy.io import wavfile

# 音频帧目录路径
AUDIO_DIR = "/Users/jinshi/odin/project/ai/LiveTalking_fork/data/dd/dd-13s-audio-frames"

class AudioFrameTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        # 获取所有音频帧文件并按文件名排序
        self.audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*")))
        self.current_index = 0
        self.start_time = None
        self.sample_rate = 16000  # 采样率
        self.frame_duration = 0.02  # 20ms 每帧
    
    async def recv(self):
        if self.start_time is None:
            self.start_time = asyncio.get_event_loop().time()
        
        # 计算当前应该播放第几帧
        elapsed_time = asyncio.get_event_loop().time() - self.start_time
        frame_no = int(elapsed_time / self.frame_duration)
        self.current_index = frame_no % len(self.audio_files)

        # 读取当前音频帧
        audio_path = self.audio_files[self.current_index]
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # 将一维数组转换为二维数组，增加一个维度
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        # 创建音频帧
        frame = AudioFrame.from_ndarray(
            audio_data,
            format='s16',
            layout='mono'
        )
        frame.sample_rate = self.sample_rate
        frame.pts = frame_no
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        # 控制帧率
        await asyncio.sleep(self.frame_duration)
        
        return frame



async def main():
    # 创建音频轨道
    audio_track = AudioFrameTrack()
    
    try:
        print("开始播放音频...")
        while True:
            frame = await audio_track.recv()
            print(f"播放音频帧 {frame.pts}, 时间戳: {frame.pts * frame.time_base}")
            
    except KeyboardInterrupt:
        print("\n停止播放")
    finally:
        audio_track.stop()

if __name__ == "__main__":
    asyncio.run(main())

