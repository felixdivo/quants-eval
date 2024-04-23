# adapted from https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Classification/src/models/gpt4ts.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from einops import rearrange, repeat
from .embed import DataEmbedding, DataEmbedding_wo_time

class GPT4ts(nn.Module):
    
    def __init__(self,max_token_length: int, max_seq_len: int,patch_size: int, stride: int, dropout: float, num_classes: int,d_model: int = 768, feat_dim: int=1):
        super().__init__()
        self.max_token_length = max_token_length
        self.seq_len = max_seq_len
        self.max_len = max_seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.gpt_layers = 6
        self.feat_dim =  feat_dim 
        self.num_classes = num_classes
        self.d_model = d_model

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding_wo_time(self.feat_dim * self.patch_size, d_model-28, dropout=dropout)
        self.text_embedding = GPT2Tokenizer.from_pretrained('gpt2')
        self.text_embedding.add_special_tokens({'pad_token': '[PAD]'})

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # device = torch.device('cuda:{}'.format(0))
        # self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(d_model * self.patch_num)
        
        self.ln_proj = nn.LayerNorm(d_model * self.patch_num)
        self.out_layer = nn.Linear(d_model * self.patch_num, self.num_classes)
        
    def forward(self, x_enc, text=None):
        B, L, M = x_enc.shape
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        
        outputs = self.enc_embedding(input_x)
        if text is not None:
            encoding = self.text_embedding(
                text,
                padding='max_length',      # Pad to the longest sequence in the batch (or set to 'max_length')
                truncation=True,   # Truncate to the max_token_length
                max_length=self.max_token_length,  # Specify the max length for truncation
                return_tensors="pt" # Return PyTorch tensors
            )["input_ids"]   
            text_embedding = repeat(encoding, 'b l -> b n l', n=outputs.size(1)).to(outputs.device)
            outputs = torch.concat((outputs, text_embedding), dim=-1) #outputs + text_embedding
        
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state


        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs
