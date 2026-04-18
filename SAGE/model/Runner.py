import os
import json
import yaml
from utils import *
from CustomDataset import CustomDataset
from Trainer import train_model, evaluate
from Sage import SAGE

def run_hatemm(config):
    dataset = 'hatemm'
    # Get data path
    train_path = config[dataset]['train_path']
    val_path = config[dataset]['val_path']
    test_path = config[dataset]['test_path']
    trans_path = config[dataset]['trans_path']
    audio_path = config[dataset]['audio_path']
    frame_path = config[dataset]['frame_path']
    n_frames = config['vision']['n_frames']
    config_train = config['train']
    train = config_train['train']
    mapping_dict = get_mapping(dataset)

    with open(trans_path, 'r') as f:
        trans_list = json.load(f)
    trans = {}
    for item in trans_list:
        video_id = item.get('Video_ID')
        if video_id:
            trans[video_id] = item

    # Load data to dataframe
    train_df = load_split_hatemm(train_path, trans, audio_path, frame_path, n_frames)
    valid_df = load_split_hatemm(val_path, trans, audio_path, frame_path, n_frames)

    device = config['device']
    # Init SAGE
    model = SAGE(config['model'])
    model.to(device)
    
    # Get Training settings
    _, _, _, _, optimizer, criterion, scheduler = get_training_setting(config, model)
    train_data = CustomDataset(config, train_df, train)
    valid_data = CustomDataset(config, valid_df, train)
    
    print(f"Train={len(train_df)}, Valid={len(valid_df)}")
    print(train_df['Label'].value_counts(), valid_df['Label'].value_counts())
    # Parameters 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train
    train_model(model, config_train, train_data, valid_data, mapping_dict, criterion, optimizer, scheduler, device)

    # Evaluate on Test Dataset
    test_df  = load_split_hatemm(test_path, trans, audio_path, frame_path, n_frames)
    test_data  = CustomDataset(config, test_df, train)


    checkpoint = torch.load(
        config_train['save_path'],
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    evaluate(model, test_data, mapping_dict, device)

def run_mhclip_bl(config, num_classes=2):
    dataset = 'mhclip_bl'
    train_path = config[dataset]['train_path']
    val_path = config[dataset]['val_path']
    test_path = config[dataset]['test_path']
    trans_path = config[dataset]['trans_path']
    audio_path = config[dataset]['audio_path']
    frame_path = config[dataset]['frame_path']
    n_frames = config['vision']['n_frames']
    config_train = config['train']
    train = config_train['train']

    mapping_dict = get_mapping(dataset, num_classes)

    with open(trans_path, 'r') as f:
        trans_list = json.load(f)
    trans = {}
    for item in trans_list:
        video_id = item.get('Video_ID')
        if video_id:
            trans[video_id] = item

    train_df = load_split_mhclip(train_path, trans, audio_path, frame_path, n_frames)
    valid_df = load_split_mhclip(val_path, trans, audio_path, frame_path, n_frames)

    device = config['device']
    model = SAGE(config['model'])
    model.to(device)
    
    _, _, _, _, optimizer, criterion, scheduler = get_training_setting(config, model)
    train_data = CustomDataset(config, train_df, train)
    valid_data = CustomDataset(config, valid_df, train)
    
    print(f"Train={len(train_df)}, Valid={len(valid_df)}")
    print(train_df['Label'].value_counts(), valid_df['Label'].value_counts())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    train_model(model, config_train, train_data, valid_data, mapping_dict, criterion, optimizer, scheduler, device)

    test_df  = load_split_mhclip(test_path, trans, audio_path, frame_path, n_frames)
    test_data  = CustomDataset(config, test_df, train)

    checkpoint = torch.load(
        config_train['save_path'],
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    evaluate(model, test_data, mapping_dict, device)
    
def run_mhclip_yt(config, num_classes=2):
    dataset = 'mhclip_yt'
    train_path = config[dataset]['train_path']
    val_path = config[dataset]['val_path']
    test_path = config[dataset]['test_path']
    trans_path = config[dataset]['trans_path']
    audio_path = config[dataset]['audio_path']
    frame_path = config[dataset]['frame_path']
    n_frames = config['vision']['n_frames']
    config_train = config['train']
    train = config_train['train']
    
    mapping_dict = get_mapping(dataset, num_classes)

    with open(trans_path, 'r') as f:
        trans_list = json.load(f)
    trans = {}
    for item in trans_list:
        video_id = item.get('Video_ID')
        if video_id:
            trans[video_id] = item

    train_df = load_split_mhclip(train_path, trans, audio_path, frame_path, n_frames, num_classes)
    valid_df = load_split_mhclip(val_path, trans, audio_path, frame_path, n_frames, num_classes)

    device = config['device']
    model = SAGE(config['model'])
    model.to(device)
    
    _, _, _, _, optimizer, criterion, scheduler = get_training_setting(config, model)
    train_data = CustomDataset(config, train_df, train)
    valid_data = CustomDataset(config, valid_df, train)
    
    print(f"Train={len(train_df)}, Valid={len(valid_df)}")
    print(train_df['Label'].value_counts(), valid_df['Label'].value_counts())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    train_model(model, config_train, train_data, valid_data, mapping_dict, criterion, optimizer, scheduler, device)

    test_df  = load_split_mhclip(test_path, trans, audio_path, frame_path, n_frames, num_classes)
    test_data  = CustomDataset(config, test_df, train)

    checkpoint = torch.load(
        config_train['save_path'],
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    evaluate(model, test_data, mapping_dict, device)
