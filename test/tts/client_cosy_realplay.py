# 调用cosy voice tts 的api

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import time
import requests
import torch
import torchaudio
import numpy as np
import sounddevice as sd  # 添加 sounddevice 导入


def main_cosy():
    url = "http://{}:{}/inference_{}".format(args.host, args.port, args.mode)
    if args.mode == 'zero_shot':
        payload = {
            'tts_text': args.tts_text,
            'prompt_text': args.prompt_text
        }
        files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
        try:
            response = requests.request("GET", url, data=payload, files=files, stream=True)
            print(f'response: {response}')
            
            # 初始化一个空的字节串用于存储音频数据
            tts_audio = b''
            # 以16000字节为块大小迭代读取响应内容
            for r in response.iter_content(chunk_size=16000):
                # 实时播放得到的音频数据
                # 跳过WAV文件头(44字节)
                if len(tts_audio) == 0:
                    r = r[44:]
                    
                # 将字节数据转换为16位整型数组
                audio_data = np.frombuffer(r, dtype=np.int16)
                
                # 使用 sounddevice 播放音频。播放每个片段，会有停顿感。
                sd.play(audio_data, samplerate=22050)
                sd.wait()  # 等待音频播放完成
                
                # 累积音频数据用于保存
                tts_audio += r
                
        except requests.exceptions.RequestException as e:
            print(f"连接服务器时出错: {e}")
            return
        except Exception as e:
            print(f"处理音频时出错: {e}")
            return

def main_local():
    url = "http://localhost:18080/audio"

    response = requests.request("GET", url, stream=True)

    print(f'response: {response}')

    # 初始化一个空的字节串用于存储音频数据
    tts_audio = b''
    # 以16000字节为块大小迭代读取响应内容
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    
    tts_audio = tts_audio[44:]

    # 将字节数据转换为16位整型数组,然后转换为PyTorch张量
    # 并在最前面添加一个维度作为batch维度
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    
    logging.info('save response to {}'.format(args.tts_wav))

    torchaudio.save(uri=args.tts_wav, src=tts_speech, sample_rate=16000)
    
    logging.info('get response')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0')
    parser.add_argument('--port',
                        type=int,
                        default='50000')
    parser.add_argument('--mode',
                        default='zero_shot',
                        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct'],
                        help='request mode')
    parser.add_argument('--tts_text',
                        type=str,
                        default='你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？')
    parser.add_argument('--spk_id',
                        type=str,
                        default='中文女')
    parser.add_argument('--prompt_text',
                        type=str,
                        default='这个星期我简直忙坏了，他对盲锣先生说。我要上观察课。',
                        # default='希望你以后能够做的比我还好呦。'
                        )
    
    parser.add_argument('--prompt_wav',
                        type=str,
                        default='test/tts/zero_shot_prompt.wav')
    parser.add_argument('--instruct_text',
                        type=str,
                        default='Theo \'Crimson\', is a fiery, passionate rebel leader. \
                                 Fights with fervor for justice, but struggles with impulsiveness.')
    parser.add_argument('--tts_wav',
                        type=str,
                        default='/Users/jinshi/Downloads/demo.wav')
    args = parser.parse_args()
    prompt_sr, target_sr = 16000, 22050

    # -- cmd: 
    #   python test/tts/client_cosy_realplay.py  --host 106.75.1.201 --tts_text "我开始说了：正值三月，春意盎然，万物复苏，正是为孩子留下学年美好纪念的绝佳时节。"
    main_cosy() 

    # main_local()


