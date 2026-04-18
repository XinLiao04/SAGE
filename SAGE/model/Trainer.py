import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score

@torch.no_grad()
def evaluate(model, eval_data, mapping_dict, device):
    dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_preds, all_labels = [], []

    total_samples = 0
    start_time = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        video_ids = batch["Video_ID"]
        text_feat = batch["text_feat"].to(device)
        audio_feat = batch["audio_feat"].to(device)
        vision_feat = batch["vision_feat"].to(device)
        labels = batch["label"].to(device)

        batch_size = labels.size(0)
        total_samples += batch_size
        
        _, _, _, logits_fusion, _, _ = model(text_feat, audio_feat, vision_feat)
        
        preds = torch.argmax(logits_fusion, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total_time = time.time() - start_time
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    macro_precision = precision_score(all_labels, all_preds, average="macro")
    macro_recall = recall_score(all_labels, all_preds, average="macro")

    print(f"Accuracy = {acc:.4f} | Macro-P = {macro_precision:.4f} | "
          f"Macro-R = {macro_recall:.4f} | Macro-F1 = {macro_f1:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=list(mapping_dict.keys()),
        output_dict=True
    )

    print("\nPer-class metrics (4 decimal places):")
    for cls_name, metrics in report.items():
        if cls_name in mapping_dict.keys():
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1-score"]
            support = metrics["support"]
            print(f"{cls_name:<10s}  P={precision:.4f} | R={recall:.4f} | F1={f1:.4f} | N={int(support)}")

    avg_time_per_sample = (total_time / total_samples) * 1000   
    print(f"\nAverage inference time per sample = {avg_time_per_sample:.4f} ms")
    return acc, macro_f1, macro_precision, macro_recall, avg_time_per_sample

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    fusion_loss_total, expert_loss_total, gate_loss_total = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        text_feat = batch["text_feat"].to(device)
        audio_feat = batch["audio_feat"].to(device)
        vision_feat = batch["vision_feat"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits_text, logits_audio, logits_vision, logits_fusion, gates, gating_mask = model(
            text_feat, audio_feat, vision_feat
        )
        
        loss, loss_spy = criterion(
            logits_text, logits_audio, logits_vision, logits_fusion, gates, gating_mask, labels
        )
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        fusion_loss_total += loss_spy.get('fusion', 0.0)
        expert_loss_total += loss_spy.get('expert', 0.0)
        gate_loss_total += loss_spy.get('gate', 0.0)
        
        batch_preds = torch.argmax(logits_fusion, dim=1)
        batch_acc = (batch_preds == labels).float().mean().item()
        total_acc += batch_acc

        all_preds.extend(batch_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    batch_num = len(train_loader)
    train_loss = total_loss / batch_num
    train_acc = total_acc / batch_num
    fusion_loss_avg = fusion_loss_total / batch_num
    expert_loss_avg = expert_loss_total / batch_num
    gate_loss_avg = gate_loss_total / batch_num

    return train_loss, train_acc, fusion_loss_avg, expert_loss_avg, gate_loss_avg, all_preds, all_labels

def collate_fn(batch):
    video_ids = [b["Video_ID"] for b in batch]
    text_feat_list = [b["text_feat"] for b in batch] 
    audio_feat_list = [b["audio_feat"] for b in batch] 
    frame_feat_list = [b["vision_feat"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], 0)

    text_feat = torch.stack(text_feat_list, 0)
    audio_feat = torch.stack(audio_feat_list, 0)
    frame_feat = torch.stack(frame_feat_list, 0)

    return {
        "Video_ID": video_ids, 
        "text_feat": text_feat,
        "audio_feat": audio_feat,
        "vision_feat": frame_feat,
        "label": labels
    }

# train
def train_model(model, config_train, train_data, valid_data, mapping_dict, criterion, optimizer, scheduler, device):
    batch_size = config_train['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    epochs = config_train['epochs']
    best_f1 = 0.0
    for epoch in range(epochs):
        train_loss, train_acc, fusion_loss, expert_loss, gate_loss, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )

        print(f"\n[Epoch {epoch+1}/{epochs}] "
              f"Train Loss={train_loss:.4f}, Fusion Loss={fusion_loss:.4f}, "
              f"Expert Loss={expert_loss:.4f}, Gate Loss={gate_loss:.4f}, "
              f"Train Acc={train_acc:.4f}")

        print('=' * 20, 'Validation Evaluation', '=' * 20)
        _, val_macro_f1, _, _, _ = evaluate(
            model, valid_loader, mapping_dict, device
        )

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, config_train['save_path'])
            print(f"Saved The Best Model (Best F1: {best_f1:.4f})")

        print('=' * 20, 'Test Evaluation', '=' * 20)
        scheduler.step() 

    print("\nTraining Complete! Best Val Macro-F1: {:.4f}".format(best_f1))