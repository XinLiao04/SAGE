import torch
import torch.nn as nn
import torch.nn.functional as F

class GED(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.output_dim = config['output_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.self_atten = nn.MultiheadAttention(embed_dim=self.output_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.cross_atten = nn.MultiheadAttention(embed_dim=self.output_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(self.output_dim)
        self.norm_sa = nn.LayerNorm(self.output_dim)
        self.norm_ffn = nn.LayerNorm(self.output_dim)
        self.ffn = nn.Sequential(
                    nn.Linear(self.output_dim, self.output_dim*4),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.output_dim*4, self.output_dim)
                )

    def getCombinedFeat(self, Core_modal, Aux_modal_1, Aux_modal_2, Core):
        if Core == 'T':
            return torch.cat([Core_modal, Aux_modal_1, Aux_modal_2], dim=1) 
        elif Core == 'A':
            return torch.cat([Aux_modal_1, Core_modal, Aux_modal_2], dim=1) 
        elif Core == 'V':
            return torch.cat([Aux_modal_1, Aux_modal_2, Core_modal], dim=1) 
        else:
            print("Core modality error. Core must be 'T', 'A' or 'V'")
            raise Exception

    def forward(self, Core_modal, Aux_modal_1, Aux_modal_2, Core):
        attn_out, _ = self.self_atten(self.norm_sa(Core_modal), self.norm_sa(Core_modal), self.norm_sa(Core_modal))
        Core_modal = Core_modal + attn_out
        
        combined_feats = self.getCombinedFeat(Core_modal, Aux_modal_1, Aux_modal_2, Core)
        attn_out, _  = self.cross_atten(self.norm_ca(Core_modal), combined_feats, combined_feats)
        Core_modal = Core_modal + attn_out
        Core_modal = Core_modal + self.ffn(self.norm_ffn(Core_modal))

        return Core_modal

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.output_dim = config['output_dim']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_dim, self.num_classes)
        )
    
    def forward(self, feat):
        return self.classifier(feat)
        
class Alignment(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.cfg = config
        self.input_dim = input_dim
        self.dropout = config['dropout']
        self.output_dim = config['output_dim']
        self.alignLayer = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.LayerNorm(self.input_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.input_dim, self.output_dim),
        )
    
    def forward(self, feat):
        return self.alignLayer(feat)

class GateNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.output_dim = config['output_dim']
        self.gate_layer = nn.Sequential(
            nn.Linear(3 * self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, 3),
            nn.Softmax(dim=-1),
        )
    
    def select_top_k_gates(self, k, gates):
        batch_size = gates.shape[0]
        scores = gates
        num_candidates = scores.shape[1]
        topk_vals, topk_indices = torch.topk(scores, k=min(k, num_candidates), dim=1)
        
        mask = torch.zeros_like(gates)
        for i in range(batch_size):
            mask[i, topk_indices[i]] = 1

        gated_gates = gates * mask
        gated_gates = gated_gates / (gated_gates.sum(dim=1, keepdim=True) + 1e-8)
        
        return gated_gates, mask
    
    def forward(self, combined_pooled_feat):
        gates = self.gate_layer(combined_pooled_feat)
        gated_gates, mask = self.select_top_k_gates(self.cfg['k'], gates)
        return gated_gates, mask

class SAGE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.output_dim = self.cfg['output_dim']
        # modality embedding 
        self.modality_emb_text = nn.Parameter(torch.randn(1, self.output_dim))
        self.modality_emb_audio = nn.Parameter(torch.randn(1, self.output_dim))
        self.modality_emb_video = nn.Parameter(torch.randn(1, self.output_dim))
        
        self.align_t = Alignment(config, config['text_dim'])
        self.align_a = Alignment(config, config['audio_dim'])
        self.align_v = Alignment(config, config['vision_dim'])

        self.classifier_text = Classifier(config)
        self.classifier_audio = Classifier(config)
        self.classifier_vision = Classifier(config)

        self.GED_t = nn.ModuleList([GED(config) for _ in range(config['num_forward'])])
        self.GED_a = nn.ModuleList([GED(config) for _ in range(config['num_forward'])])
        self.GED_v = nn.ModuleList([GED(config) for _ in range(config['num_forward'])])

        self.gate_layer = GateNetwork(config)

    def forward(self, text_input, audio_input, vision_input):
        B = text_input.size(0)
        # proj 
        text_input = self.align_t(text_input) 
        audio_input = self.align_a(audio_input)
        vision_input = self.align_v(vision_input) 

        # modality embedding
        T = text_input + self.modality_emb_text.unsqueeze(0).expand(B, -1, -1)
        A = audio_input + self.modality_emb_audio.unsqueeze(0).expand(B, -1, -1)
        V = vision_input + self.modality_emb_video.unsqueeze(0).expand(B, -1, -1)

        for i in range(self.cfg['num_forward']):
            T = self.GED_t[i](T, A, V, 'T')
            A = self.GED_a[i](A, T, V, 'A')
            V = self.GED_v[i](V, T, A, 'V')
        
        #pool Text
        pooled_T = T.mean(dim = 1)
        # mask pool audio 
        mask = (A.abs().sum(-1) != 0).float()
        audio_sum = (A * mask.unsqueeze(-1)).sum(dim=1)
        audio_len = mask.sum(dim=1, keepdim=True)
        pooled_A = audio_sum / audio_len  # [B, D]
        # pool vision
        pooled_V = V.mean(dim = 1)

        pool_concat = torch.cat([pooled_T, pooled_A, pooled_V], dim=-1)
        gates, gating_mask = self.gate_layer(pool_concat)
        g_T, g_A, g_V = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        

        logits_text = self.classifier_text(pooled_T)
        logits_audio = self.classifier_audio(pooled_A)
        logits_vision = self.classifier_vision(pooled_V)
        logits_fusion = g_T * logits_text + g_A * logits_audio + g_V * logits_vision
        
        return logits_text, logits_audio, logits_vision, logits_fusion, gates, gating_mask