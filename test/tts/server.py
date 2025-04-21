"""
建立一个server，收到请求后，将音频 “/Users/jinshi/odin/project/ai/metahuman-stream/dd/dd-13s.wav” 转换成byte，通过流式的方式返回
"""

import os
from aiohttp import web
import aiohttp
import asyncio

# 音频文件路径
AUDIO_FILE = "/Users/jinshi/odin/project/ai/metahuman-stream/dd/dd-13s.wav"

async def stream_audio(request):
    """以流式方式返回音频数据"""
    if not os.path.exists(AUDIO_FILE):
        return web.Response(status=404, text="音频文件不存在")

    response = web.StreamResponse()
    response.headers['Content-Type'] = 'audio/wav'
    
    try:
        await response.prepare(request)
        
        # 以二进制方式读取音频文件
        chunk_size = 1024 * 8  # 8KB chunks
        with open(AUDIO_FILE, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                try:
                    await response.write(chunk)
                    await asyncio.sleep(0.01)  # 控制发送速率
                except ConnectionResetError:
                    # 客户端断开连接，避免客户端主动断开连接报错
                    return response
                except Exception as e:
                    # 其他错误
                    print(f"发送数据时出错: {str(e)}")
                    return response
            
        await response.write_eof()
        return response
        
    except Exception as e:
        print(f"流式传输出错: {str(e)}")
        return response

async def index(request):
    """返回一个简单的HTML页面用于测试"""
    html = """
        <html>
            <body>
                <h1>音频流测试</h1>
                <audio controls>
                    <source src="/audio" type="audio/wav">
                    您的浏览器不支持音频播放
                </audio>
            </body>
        </html>
    """
    return web.Response(text=html, content_type='text/html')

app = web.Application()
app.router.add_get('/', index)
app.router.add_get('/audio', stream_audio)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=18080)

# url: http://127.0.0.1:18080/audio
