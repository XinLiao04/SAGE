import os
import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import math
from CustomLoss import CustomLoss

def get_mapping(dataset, num_classes=2):
    if dataset == 'hatemm':
        return {"Non Hate": 0, "Hate" : 1}
    elif dataset.startswith('mhclip'):
        if num_classes == 2:
            return {"Normal": 0, "Hateful" : 1}
        elif num_classes == 3:
            return  {"Normal": 0, "Offensive": 1, "Hateful" : 2}
    else:
        print('Error Dataset Name.')
        raise Exception

def load_split_hatemm(data_path, trans, audio_dir, frame_dir, n_frames):
    df = pd.read_csv(data_path)
    df = df.fillna('')
    mapping_dict = get_mapping('hatemm')

    samples = []
    for _, row in df.iterrows():
        video_id = row["video_file_name"]
        video_id = video_id[:-4]
        # text
        text_info = trans.get(video_id, {})
        text = str(text_info.get('Transcript', ""))
        # label
        label = row["label"]
        # audio
        video_path = row['video_path']
        audio_path = f"{audio_dir}/{video_id}.wav"
        # frames
        frame_paths = [os.path.join(f"{frame_dir}", f"{video_id}_{i}.jpg") for i in range(1, n_frames + 1)]

        if all(os.path.exists(p) for p in frame_paths) and os.path.exists(audio_path):
            samples.append((video_id, text, audio_path, frame_paths, label))
    
    df_out = pd.DataFrame(samples, columns=['Video_ID', 'Text', 'AudioPath', 'FramePaths',  'Label'])
    df_out['Label'] = df_out['Label'].map(mapping_dict)
    return df_out

def load_split_mhclip(tsv_path, trans, audio_dir, frame_dir, n_frames, num_classes = 2):
    df = pd.read_csv(tsv_path, sep='\t')

    mapping_dict = get_mapping('mhclip', num_classes)
    samples = []
    for _, row in df.iterrows():
        video_id = row["Video_ID"]
        # text
        text_info = trans.get(video_id, {})
        title = text_info.get('Title', '')
        transcript = text_info.get('Transcript', '')
        description = text_info.get('description', '')
        text = f"{title}\n{description}\n{transcript}"
        # label
        majority_voting = row["Majority_Voting"]
        labels = row["Label"]
        # audio
        audio_path = f"{audio_dir}/{video_id}.wav"
        # frames
        frame_paths = [os.path.join(f"{frame_dir}", f"{video_id}_{i}.jpg") for i in range(1, n_frames + 1)]
        if all(os.path.exists(p) for p in frame_paths) and os.path.exists(audio_path):
            samples.append((video_id, text, audio_path, frame_paths, majority_voting))
    
    df_out = pd.DataFrame(samples, columns=['Video_ID', 'Text', 'AudioPath', 'FramePaths',  'Label'])
    df_out['Label'] = df_out['Label'].map(mapping_dict)
    return df_out

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        # linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_training_setting(config, model):
    config_train = config['train']
    lr = config_train['lr']
    epochs = config_train['epochs']
    batch_size = config_train['batch_size']
    warmup_steps = config_train['warmup_steps']
    if config_train['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=float(config_train['weight_decay']), lr=float(lr))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = CustomLoss(config['loss'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs
    )
    return lr, epochs, batch_size, warmup_steps, optimizer, criterion, scheduler