import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask:[B,1,L,L]
        att = att.masked_fill(mask == 0, float('-inf'))

        if x.dtype == torch.float16:
            att = att.to(torch.float32)
            fp16 = True
        else:
            fp16 = False
        att = F.softmax(att, dim=-1)
        if fp16:
            att = att.to(torch.float16)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            # nn.GELU(),  # nice, GELU is not valid in torch<1.6
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTPrediction(nn.Module):
    def __init__(self, config):
        super(GPTPrediction, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = GELU()
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x


class PoseMLP(nn.Module):
    def __init__(self, embedding_dim, pose_dim):
        super(PoseMLP, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x B*13*3
        x = self.block1(x)
        x = self.block2(x)
        out = self.block3(x)
        return out


class GPTPose(nn.Module):
    def __init__(self, config, pose_dim=3):
        super().__init__()
        # input embedding stem
        embedding_dim = config.n_embd if not config.init_gpt_with_vqvae else config.n_quant
        self.z_emb = nn.Embedding(config.vocab_size, embedding_dim)
        self.z_sp = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, config.n_embd)))  # 分隔符(用于预测第一个patch)
        self.c_emb = nn.Embedding(config.vocab_size, embedding_dim)
        self.c_sp = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, config.n_embd)))
        self.pos_emb = nn.Embedding(config.sequence_length, config.n_embd)
        self.pose_mlp = PoseMLP(config.n_embd, pose_dim)
        if config.init_gpt_with_vqvae:
            self.emb_proj = nn.Linear(embedding_dim, config.n_embd)
        else:
            self.emb_proj = None
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.dec = GPTPrediction(config)
        if config.init_gpt_with_vqvae:
            self.dec_proj = nn.Linear(config.n_embd, embedding_dim)
        else:
            self.dec_proj = None
        # z dec
        self.z_dec = nn.Linear(self.z_emb.weight.size(1), self.z_emb.weight.size(0), bias=False)
        self.z_dec.weight = self.z_emb.weight
        self.z_bias = nn.Parameter(torch.zeros(self.z_emb.weight.size(0)))
        # c dec
        self.c_dec = nn.Linear(self.c_emb.weight.size(1), self.c_emb.weight.size(0), bias=False)
        self.c_dec.weight = self.c_emb.weight
        self.c_bias = nn.Parameter(torch.zeros(self.c_emb.weight.size(0)))

        self.sequence_length = config.sequence_length
        self.apply(self._init_weights)
        self.config = config

    def forward(self, c_idx, z_idx, src_pose, tgt_pose, mask=None):
        # forward the GPT model
        c_embeddings = self.c_emb(c_idx)
        if self.config.init_gpt_with_vqvae:
            c_embeddings = self.emb_proj(c_embeddings)
        c_embeddings = torch.cat([self.c_sp.unsqueeze(0).repeat(c_idx.shape[0], 1, 1), c_embeddings], dim=1)
        if src_pose is not None:
            src_pose_emb = self.pose_mlp(src_pose)  # [B,13,D]
            c_embeddings = torch.cat([src_pose_emb, c_embeddings], dim=1) # [B,L+13,D]

        z_embeddings = self.z_emb(z_idx)
        if self.config.init_gpt_with_vqvae:
            z_embeddings = self.emb_proj(z_embeddings)
        z_embeddings = torch.cat([self.z_sp.unsqueeze(0).repeat(z_idx.shape[0], 1, 1), z_embeddings], dim=1)
        tgt_pose_emb = self.pose_mlp(tgt_pose)  # [B,13,D]
        z_embeddings = torch.cat([tgt_pose_emb, z_embeddings], dim=1)  # [B,L+13,D]

        tc = c_embeddings.shape[1]
        tz = z_embeddings.shape[1]
        tok_embeddings = torch.cat([c_embeddings, z_embeddings], dim=1)
        assert tc + tz <= self.sequence_length, "Cannot forward, model sequence length is exhausted."
        position_ids = torch.arange(tc + tz, dtype=torch.long, device=z_idx.device)  # [L,]
        position_ids = position_ids.unsqueeze(0).repeat(z_idx.shape[0], 1)
        position_embeddings = self.pos_emb(position_ids)
        x = self.drop(tok_embeddings + position_embeddings)
        mask = mask.unsqueeze(1)  # [B,1,2(L+13),2(L+13)]
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.dec(x)
        if self.config.init_gpt_with_vqvae:
            x = self.dec_proj(x)
        xc = x[:, :tc, :]
        xz = x[:, tc:tc + tz, :]
        logits_c = self.c_dec(xc) + self.c_bias
        logits_z = self.z_dec(xz) + self.z_bias

        return logits_c, logits_z

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
