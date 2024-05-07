# adapted from https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Classification/src/models/gpt4ts.py
from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from einops import rearrange, repeat
from .embed import DataEmbedding, DataEmbedding_wo_time, DataEmbeddingTextConcat

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
        self.enc_embedding = DataEmbeddingTextConcat(self.patch_size, d_model, dropout=dropout)
        self.text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_tokenizer.pad_token_id = self.text_tokenizer.eos_token_id

        self.gpt2 = cast(GPT2Model,GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)).train(True)

        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        # # # self.text_embedding = nn.Embedding(self.text_tokenizer.vocab_size + 1, self.d_model)
        # # self.text_embedding.weight.data[:self.text_tokenizer.vocab_size] = self.gpt2.wte.weight.data
        # self.text_embedding.weight.data[self.text_tokenizer.pad_token_id] = torch.randn(768) * 0.01 #torch.zeros(768)

        
        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # device = torch.device('cuda:{}'.format(0))
        # self.gpt2.to(device=device)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(d_model * self.patch_num * 77)
        # self.ln_proj = nn.LayerNorm(d_model)
        # self.ln_proj = nn.Lazy
        
        self.out_layer = nn.Linear(d_model , self.num_classes, bias=False)
        # self.out_layer = nn.Linear(d_model , self.num_classes, bias=False)
        
    def forward(self, x_enc, text=None):
        B, L, M = x_enc.shape
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        # check for nan
        # if torch.isnan(input_x).any():
        #     print('nan in input_x')
        #     # find where nan in dim 1
        #     print(torch.isnan(input_x).any(dim=1))

        # input_x = input_x[:,:40]
        input_x = rearrange(input_x, 'b m n p -> b (n m) p') # interleave variate patches

        text_tok = self.text_tokenizer(text, padding='max_length', truncation=True, max_length=self.max_token_length, return_tensors="pt")["input_ids"].to(input_x.device)
        text_embed = self.gpt2.wte(text_tok)

        outputs = self.enc_embedding(input_x, text_embed)
        output = self.act(self.gpt2(inputs_embeds=outputs).last_hidden_state)
        # output = output.view(B, -1) 
        # output = self.ln_proj(output)       
        output = self.out_layer(output)
        logits = output[:, -1]


        # outputs = torch.zeros(B, input_x.shape[1], self.patch_num, self.d_model).to(input_x.device)
        # outputs_lst = [] 
        # for i in range(input_x.size(1)):
        #     outputs[:, i] = self.enc_embedding(input_x[:, i])
        #     outputs[:,i] = self.act(self.gpt2(inputs_embeds=outputs[:, i]).last_hidden_state)
        #     output = self.out_layer(outputs[:, i])
        #     logits = output[:, -1]
        #     outputs_lst.append(logits)
        
        
        # majority vote for classification of all variates
        # res = torch.stack(outputs_lst, dim=1)
        # logits = torch.mean(res, dim=1)



        # outputs = self.enc_embedding(input_x)
        # if text is not None:
        #     encoding = self.text_embedding(
        #         text,
        #         padding='max_length',      # Pad to the longest sequence in the batch (or set to 'max_length')
        #         truncation=True,   # Truncate to the max_token_length
        #         max_length=self.max_token_length,  # Specify the max length for truncation
        #         return_tensors="pt" # Return PyTorch tensors
        #     )["input_ids"]   
        #     text_embedding = repeat(encoding, 'b l -> b n l', n=outputs.size(1)).to(outputs.device)
        #     outputs = torch.concat((outputs, text_embedding), dim=-1) #outputs + text_embedding
        
        # outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        # outputs = self.act(outputs)


        # outputs = self.act(outputs).reshape(B, -1)
        
        # outputs = self.ln_proj(outputs)
        # outputs = self.out_layer(outputs)
        # logits = outputs[:, -1]
        
        return logits
