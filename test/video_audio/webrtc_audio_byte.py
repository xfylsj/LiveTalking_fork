"""
使用webrtc协议，将"/Users/jinshi/odin/project/ai/metahuman-stream/dd/dd-13s-video-frames"目录下的图片，作为每一个视频帧，推送到前端html页面。
同时将"/Users/jinshi/odin/project/ai/metahuman-stream/dd/dd-13s-audio-frames"目录下的音频，推送到前端html页面。
要求：
1. 使用webrtc协议，将图片按照文件名顺序推送到前端html页面，所有文件推送完成后，循环从头再次推送。
2. 制作html页面，页面上有一个开始按钮，点击以后开始接收视频帧和音频，并播放
"""
import fractions
import json
import os
import aiohttp
import cv2
import asyncio
import glob
import numpy as np
from aiohttp import web
from av import VideoFrame, AudioFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import argparse  # 添加到文件顶部的导入部分

# 图片目录路径
IMG_DIR = "/Users/jinshi/odin/project/ai/metahuman-stream/dd/dd-13s-video-frames"

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="WebRTC音视频流服务器")
    parser.add_argument("--audio_url", 
                       type=str, 
                       default="http://localhost:18080/audio",
                       help="音频服务器URL")
    parser.add_argument("--host", 
                       type=str, 
                       default="0.0.0.0",
                       help="服务器监听地址")
    parser.add_argument("--port", 
                       type=int, 
                       default=18081,
                       help="服务器监听端口")
    return parser.parse_args()

class VideoImageTrack(MediaStreamTrack):
    kind = "video"
    
    def __init__(self):
        super().__init__()
        # 获取所有图片文件并按文件名排序
        self.image_files = sorted(glob.glob(os.path.join(IMG_DIR, "*")))
        self.current_index = 0
        self.start_time = None
    
    # 每当 WebRTC 需要发送一帧视频，就会调用 recv()
    async def recv(self):
        if self.start_time is None:
            self.start_time = asyncio.get_event_loop().time()
        
        # 计算当前应该显示第几帧
        elapsed_time = asyncio.get_event_loop().time() - self.start_time
        frame_no = int(elapsed_time * 30)  # 30fps
        self.current_index = frame_no % len(self.image_files)

        # 读取当前图片
        img_path = self.image_files[self.current_index]
        frame = cv2.imread(img_path)
        if frame is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为VideoFrame
        video_frame = VideoFrame.from_ndarray(frame)
        video_frame.pts = frame_no

        try:
            video_frame.time_base = fractions.Fraction(1, 30)  # 使用分数形式表示 1/30
        except Exception as e:
            print(f"设置 time_base 时出错: {e}")
        
        # 控制帧率
        await asyncio.sleep(1/30)
        
        return video_frame

class AudioFrameTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, audio_url):
        super().__init__()
        self.audio_url = audio_url
        self.sample_rate = 16000
        self.chunk_size = 320 * 2  # 改为固定大小：320采样点 * 2字节
        self.session = None
        self.response = None
        self.buffer = b''
        
    async def init_connection(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        self.response = await self.session.get(self.audio_url)
        # 跳过WAV文件头(44字节)
        await self.response.content.read(44)
    
    async def stop(self):
        """清理资源"""
        if self.response is not None:
            self.response.close()
        if self.session is not None:
            await self.session.close()
            self.session = None
        await super().stop()
        
    async def recv(self):
        try:
            if self.session is None:
                await self.init_connection()
                
            # 读取音频数据
            chunk = await self.response.content.read(self.chunk_size)
            if not chunk:
                # 如果读完了,重新开始
                await self.stop()  # 先清理旧的连接
                await self.init_connection()  # 重新建立连接
                chunk = await self.response.content.read(self.chunk_size)
                
            # 确保数据长度正确
            if len(chunk) < self.chunk_size:
                # 如果数据不足，用静音补齐
                chunk = chunk + b'\x00' * (self.chunk_size - len(chunk))
                
            # 将bytes转换为numpy数组
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            
            # 创建音频帧
            frame = AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),  # 修改为 (1, 320) 的形状
                format="s16",  # 16位有符号整数
                layout="mono"  # 单声道
            )
            frame.sample_rate = self.sample_rate
            frame.pts = int(asyncio.get_event_loop().time() * self.sample_rate)
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            
            # 控制帧率
            await asyncio.sleep(0.02)
            
            return frame
            
        except Exception as e:
            print(f"音频接收错误: {e}")
            # 发生错误时清理资源
            await self.stop()
            raise

async def index(request):
    content = """
    <html>
    <head>
        <title>WebRTC Stream</title>
        <style>
            #video {
                width: 640px;
                height: 480px;
                margin: 20px;
                background: #000;
                display: none;
            }
            #startButton, #stopButton {
                font-size: 18px;
                padding: 10px 20px;
                margin: 20px;
                cursor: pointer;
            }
            #stopButton {
                display: none;
            }
            .error {
                color: red;
                margin: 20px;
            }
            .status {
                margin: 20px;
            }
            #videoContainer {
                width: 640px;
                height: 480px;
                margin: 20px;
                background: #000;
            }
        </style>
    </head>
    <body>
        <button id="startButton">开始播放</button>
        <button id="stopButton">停止播放</button>
        <div id="videoContainer">
            <video id="video" autoplay playsinline></video>
        </div>
        <div id="status" class="status"></div>
        <div id="error" class="error"></div>
        <script>
            let pc = null;
            
            async function start() {
                try {
                    if(pc) {
                        console.log('WebRTC连接已存在');
                        return;
                    }
                    
                    const status = document.getElementById('status');
                    const video = document.getElementById('video');
                    const startButton = document.getElementById('startButton');
                    const stopButton = document.getElementById('stopButton');
                    
                    status.textContent = '正在建立连接...';
                    console.log('开始创建WebRTC连接...');
                    
                    // 显示视频元素和停止按钮，隐藏开始按钮
                    video.style.display = 'block';
                    startButton.style.display = 'none';
                    stopButton.style.display = 'inline-block';
                    
                    // 创建WebRTC连接
                    pc = new RTCPeerConnection({
                        iceServers: [
                            {urls: ['stun:stun.l.google.com:19302']}
                        ]
                    });
                    
                    // 监听ICE连接状态
                    pc.oniceconnectionstatechange = () => {
                        console.log('ICE连接状态:', pc.iceConnectionState);
                        status.textContent = 'ICE连接状态: ' + pc.iceConnectionState;
                    };
                    
                    // 监听连接状态
                    pc.onconnectionstatechange = () => {
                        console.log('连接状态:', pc.connectionState);
                        status.textContent = '连接状态: ' + pc.connectionState;
                    };
                    
                    // 添加视频和音频收发器
                    pc.addTransceiver('video', {direction: 'recvonly'});
                    pc.addTransceiver('audio', {direction: 'recvonly'});
                    
                    // 处理媒体轨道
                    pc.ontrack = function(event) {
                        if (event.track.kind === 'video') {
                            console.log('收到视频轨道');
                            const videoElem = document.getElementById('video');
                            videoElem.srcObject = event.streams[0];
                            
                            // 确保视频能播放
                            videoElem.onloadedmetadata = () => {
                                console.log('视频元数据已加载');
                                videoElem.play().catch(e => console.error('视频播放失败:', e));
                            };
                        }
                    };
                    
                    // 创建并发送offer
                    const offer = await pc.createOffer();
                    await pc.setLocalDescription(offer);
                    
                    console.log('发送offer到服务器...');
                    const response = await fetch('/offer', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            sdp: pc.localDescription.sdp,
                            type: pc.localDescription.type
                        })
                    });
                    
                    const answer = await response.json();
                    console.log('收到服务器answer');
                    await pc.setRemoteDescription(answer);
                    
                } catch(e) {
                    console.error('错误:', e);
                    document.getElementById('error').textContent = '连接失败: ' + e.message;
                    if(pc) {
                        pc.close();
                        pc = null;
                    }
                }
            }
            
            async function stop() {
                if (pc) {
                    pc.close();
                    pc = null;
                }
                const video = document.getElementById('video');
                const startButton = document.getElementById('startButton');
                const stopButton = document.getElementById('stopButton');
                
                video.srcObject = null;
                video.style.display = 'none';
                startButton.style.display = 'inline-block';
                stopButton.style.display = 'none';
                document.getElementById('status').textContent = '已停止播放';
            }
            
            document.getElementById('startButton').onclick = start;
            document.getElementById('stopButton').onclick = stop;
        </script>
    </body>
    </html>
    """
    return web.Response(content_type="text/html", text=content)

# 修改 offer 函数，接收 audio_url 参数
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"]
    )
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    # 创建音频轨道
    audio_sender = AudioFrameTrack(app['audio_url'])
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"连接状态: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            try:
                await audio_sender.stop()  # 确保使用 await
            except Exception as e:
                print(f"停止音频轨道时出错: {e}")
            await pc.close()
            pcs.discard(pc)
    
    video_sender = VideoImageTrack()
    pc.addTrack(video_sender)
    pc.addTrack(audio_sender)
    
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

pcs = set()
app = web.Application()
# 将参数存储在应用程序状态中
app['audio_url'] = parse_args().audio_url

app.router.add_get("/", index)
app.router.add_post("/offer", offer)

if __name__ == "__main__":
    args = parse_args()
    
    web.run_app(app, host=args.host, port=args.port)
