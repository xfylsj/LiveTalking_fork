
"""
从指定的视频文件中，提取出每一视频帧，分离音频，并保存到指定的目录下。
视频文件位置：/Users/jinshi/odin/project/ai/metahuman-stream/dd/dd.mp4
视频帧位置：/Users/jinshi/odin/project/ai/metahuman-stream/dd/frames
音频帧位置：/Users/jinshi/odin/project/ai/metahuman-stream/dd/audio
"""
import cv2
import os
from pydub import AudioSegment

# 定义输入输出路径
video_path = "/Users/jinshi/odin/project/ai/metahuman-stream/dd/dd.mp4"
video_output_dir = "/Users/jinshi/odin/project/ai/metahuman-stream/dd/frames"
audio_output_dir = "/Users/jinshi/odin/project/ai/metahuman-stream/dd/audio"

# 确保输出目录存在
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(audio_output_dir, exist_ok=True)

# 提取音频
audio = AudioSegment.from_file(video_path)
audio.export(os.path.join(audio_output_dir, "audio.wav"), format="wav")

# 打开视频文件提取视频帧
cap = cv2.VideoCapture(video_path)

# 获取视频的基本信息
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频帧率: {fps}")
print(f"总帧数: {frame_count}")

# 读取并保存每一帧
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 保存帧,使用6位数字命名,保证顺序
    frame_name = f"{frame_idx:06d}.jpg"
    frame_path = os.path.join(video_output_dir, frame_name)
    cv2.imwrite(frame_path, frame)
    
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"已处理 {frame_idx} 帧")

# 释放资源
cap.release()
print(f"完成! 共提取了 {frame_idx} 帧视频图像和音频")
