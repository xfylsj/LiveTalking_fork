"""
prompt:
访问api：http://localhost:18080/audio，通过流式的方式得到byte类型的音频数据，并实时播放所得到的音频
"""

import aiohttp
import asyncio
import pyaudio
import sys


async def play_audio_stream(server_url="http://localhost:18080/audio"):
    # 初始化PyAudio
    p = pyaudio.PyAudio()
    
    # 设置音频参数
    CHUNK = 1024 * 8  # 每次读取的块大小
    FORMAT = pyaudio.paInt16  # 16位格式
    CHANNELS = 1  # 单声道
    RATE = 16000  # 采样率
    
    # 打开音频流
    stream = p.open(format=FORMAT,
                   channels=CHANNELS, 
                   rate=RATE,
                   output=True)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server_url) as response:
                print("开始接收音频流...")
                
                # 跳过WAV文件头(44字节)
                header = await response.content.read(44)

                print(f'header = {header}')
                
                # 循环读取并播放音频数据
                while True:
                    chunk = await response.content.read(CHUNK)
                    if not chunk:
                        break
                    else:
                        print(f'current chuck = {chunk}')
                    
                    # 播放音频数据
                    stream.write(chunk)
                    
                print("音频播放完成")
                
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 关闭音频流和PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    # 可以通过命令行参数传入完整的服务器URL
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18080/audio"
    asyncio.run(play_audio_stream(server_url))
