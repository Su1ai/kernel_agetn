import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# Scaled-down MiniGPT transformer block (KernelBench-style problem).


class NewGELU(nn.Module):
    """GELU activation (identical to OpenAI GPT / Google BERT reference)."""

    def __init__(self):
        super(NewGELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Vanilla multi-head masked self-attention with output projection."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                             .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Model(nn.Module):
    """A single minGPT transformer block (attention + MLP + layernorm + residual)."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


# Dimensions and dtype retuned for the PPU accelerator.
# The original 4096-wide fp32 block exceeded the PPA fp32 matmul throughput and
# saturated attention softmax, so the block is narrowed and run in bf16.
# Original KernelBench level3/44 uses: batch=128, seq=512, n_embd=768, n_head=8.
batch_size = 1
max_seqlen = 64
seq_len = 64
n_embd = 1024
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

# Single source of truth for the activation dtype. test.py mirrors this so the
# reference and the Triton kernel are compared at equal precision.
dtype = torch.bfloat16


def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd, dtype=dtype)]


def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
