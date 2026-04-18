import os
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.ce = nn.CrossEntropyLoss()
        self.lambda_expert = config['lambda_expert']
        self.lambda_gate = config['lambda_gate']
    
    def compute_moe_loss(self, gates, mask):
        batch_size = gates.shape[0]

        total_gate = gates.sum(dim=0)
        total_load = mask.float().sum(dim=0)

        gate_ratio = total_gate / batch_size
        load_ratio = total_load / batch_size

        moe_loss = torch.sum(gate_ratio * load_ratio)
        return moe_loss

    def forward(self, logits_text, logits_audio, logits_vision, logits_fusion, gating, gating_mask, labels):
        fusion_loss = self.ce(logits_fusion, labels)
        expert_loss = (self.ce(logits_text, labels) + self.ce(logits_audio, labels) + self.ce(logits_vision, labels)) / 3.0
        gate_loss = self.compute_moe_loss(gating, gating_mask)

        total = fusion_loss + self.lambda_expert * expert_loss + self.lambda_gate * gate_loss
        return total, {"fusion": fusion_loss.item(), "expert": expert_loss.item(), "gate": gate_loss.item()}