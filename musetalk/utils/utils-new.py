import os
import cv2
import numpy as np
import torch

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

    
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding

def load_all_model__origin():
    audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    vae = VAE(model_path = "./models/sd-vae-ft-mse/")
    unet = UNet(unet_config="./models/musetalk/musetalk.json",
                model_path ="./models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return audio_processor,vae,unet,pe

def load_all_model(
    feature_extractor_path='./models/whisper',
    unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
    vae_type="sd-vae",
    unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
):
    audio_processor = AudioProcessor(feature_extractor_path=feature_extractor_path) 
    vae = VAE(model_path = os.path.join("models", vae_type),)
    print(f"load unet model from {unet_model_path}")
    unet = UNet(
        unet_config=unet_config,
        model_path=unet_model_path,
    )
    pe = PositionalEncoding(d_model=384)
    return audio_processor, vae, unet, pe


def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def datagen(whisper_chunks,
            vae_encode_latents,
            batch_size=8,
            delay_frame=0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, latent_batch

def load_audio_model():
    audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    return audio_processor

def load_diffusion_model():
    vae = VAE(model_path = "./models/sd-vae-ft-mse/")
    unet = UNet(unet_config="./models/musetalk/musetalk.json",
                model_path ="./models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return vae,unet,pe
