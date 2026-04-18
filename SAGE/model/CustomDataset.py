import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import VideoMAEFeatureExtractor, VideoMAEModel, AutoTokenizer, AutoModel
import torchaudio
from torchaudio.transforms import MFCC, MelSpectrogram
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.io import read_image
import pandas as pd
import numpy as np
import random

class VideoTransform:
    def __init__(self, config, train=True):
        self.cfg = config
        self.image_size = self.cfg['image_size']
        self.scale = self.cfg['scale']
        self.hflip_p = self.cfg['hflip_p']
        self.mean = self.cfg['mean']
        self.std = self.cfg['std']
        self.train = train

    def __call__(self, frames):
    # frames: list of PIL.Image or Tensor (T,C,H,W)

        # Resize
        frames = [F.resize(frame, (self.image_size, self.image_size)) for frame in frames]
        if self.train:
            # RandomResizedCrop
            i, j, h, w = T.RandomResizedCrop.get_params(frames[0], scale=self.scale, ratio=(3/4, 4/3))
            frames = [F.resized_crop(frame, i, j, h, w, (self.image_size, self.image_size)) for frame in frames]

            # RandomHorizontalFlip
            if random.random() < self.hflip_p:
                frames = [F.hflip(frame) for frame in frames]

        # Normalize
        frames = [(frame - torch.tensor(self.mean)[:,None,None]) / torch.tensor(self.std)[:,None,None] for frame in frames]

        # Stack to Tensor (T,C,H,W)
        frames = torch.stack(frames)
        return frames

class VisionEncoder(nn.Module):
    def __init__(self, config, train):
        super().__init__()
        self.device = config['device']
        self.cfg = config['vision']
        self.train = train
        self.model = VideoMAEModel.from_pretrained(self.cfg['model_name']).eval().to(self.device)
        self.video_transform = VideoTransform(self.cfg, self.train)
    
    def forward(self, frames):
        frames = self.video_transform(frames).unsqueeze(0)  # (T, C, H, W)
        frames = frames.to(self.device)
        with torch.no_grad():
            outputs = self.model(frames)
            vision_feature = outputs.last_hidden_state.squeeze(0)

        return vision_feature
        
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.cfg = config['text']
        self.max_length = self.cfg['max_length']
        self.model_name = self.cfg['model_name']
        self.model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def forward(self, text):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_outputs = self.model(**encoded)
            text_feature = text_outputs.last_hidden_state.squeeze(0)
    
        return text_feature

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.cfg = config['audio']
        self.mfcc_transform = MFCC(
            sample_rate=self.cfg['sr'],
            n_mfcc=self.cfg['n_mfcc'],
            melkwargs={
                "n_fft": self.cfg['n_fft'],
                "hop_length": self.cfg['hop_length'],
                "n_mels": self.cfg['n_mels'],
                "mel_scale": "htk",
            }
        ).to(self.device)
        self.max_audio_len = self.cfg['max_audio_len']

    def forward(self, audio_path):
        try:
            waveform, audio_sr = torchaudio.load(audio_path)  # shape: [channels, time]
            if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.to(self.device)
            mfcc = self.mfcc_transform(waveform)
            audio_feature = mfcc.squeeze(0).T  

            current_len = audio_feature.size(0)
            if current_len > self.max_audio_len:
                audio_feature = audio_feature[:self.max_audio_len, :]
            elif current_len < self.max_audio_len:
                pad_len = self.max_audio_len - current_len
                pad = torch.zeros(
                    (pad_len, audio_feature.size(1)),
                    device=audio_feature.device,
                    dtype=audio_feature.dtype
                )
                audio_feature = torch.cat([audio_feature, pad], dim=0)
        except Exception as e:
            print(f"[Warning] Audio load failed at {audio_path}: {e}")
            audio_feature = torch.zeros((self.cfg['max_audio_len'], self.cfg['n_mfcc']), dtype=torch.float32).to(self.device)

        return audio_feature

class CustomDataset(Dataset):
    def __init__(self, config, data, train):
        self.data = data
        self.t_model = TextEncoder(config)
        self.a_model = AudioEncoder(config)
        self.v_model = VisionEncoder(config, train)
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row['Video_ID']
        text = row["Text"]
        audio_path = row["AudioPath"]
        frame_path = row["FramePaths"]
        label = torch.tensor(self.data.iloc[idx]["Label"], dtype=torch.long)
        
        # text feature
        text_feature = self.t_model(text)

        # audio feature
        audio_feature = self.a_model(audio_path)

        # vision feature
        frames = [read_image(p) for p in frame_path]
        vision_feature = self.v_model(frames)

        return {
            "Video_ID": video_id,
            "text_feat": text_feature,
            "audio_feat": audio_feature, 
            "vision_feat": vision_feature,
            "label": label
        }