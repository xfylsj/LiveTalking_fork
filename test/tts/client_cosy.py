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


def main_cosy():
    url = "http://{}:{}/inference_{}".format(args.host, args.port, args.mode)
    if args.mode == 'zero_shot':
        payload = {
            'tts_text': args.tts_text,
            'prompt_text': args.prompt_text
        }
        files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
        response = requests.request("GET", url, data=payload, files=files, stream=True)

        print(f'response: {response}')

    
    # 初始化一个空的字节串用于存储音频数据
    tts_audio = b''
    # 以16000字节为块大小迭代读取响应内容
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r

        print(f'current stream data = {r[:10]}')

    # 将字节数据转换为16位整型数组,然后转换为PyTorch张量
    # 并在最前面添加一个维度作为batch维度
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    
    logging.info('save response to {}'.format(args.tts_wav))
    torchaudio.save(args.tts_wav, tts_speech, target_sr)
    logging.info('get response')

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
    main_cosy()
    # main_local()


