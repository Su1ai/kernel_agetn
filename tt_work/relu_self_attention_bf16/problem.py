import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# KernelBench level3/50: a multi-head masked self-attention layer that uses
# ReLU instead of Softmax.  Dimensions and dtype are retuned for the accelerator,
# so the layer is narrowed and run in bf16 (mirrors tt_work/triton_04_minigpt).


class Model(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that
    uses ReLU instead of Softmax.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in
        # the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(
                1, 1, max_seqlen, max_seqlen
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.relu(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs
        return y


# Dimensions and dtype retuned for the accelerator (option 2): the layer is
# narrowed and run in bf16.  Original KernelBench level3/50 uses
# batch=16, max_seqlen=1024, n_embd=768, n_head=12.
batch_size = 4
max_seqlen = 64
seq_len = 64
n_embd = 2048
n_head = 8

# Single source of truth for the activation dtype. test.py mirrors this so the
# reference and the Triton kernel are compared at equal precision.
dtype = torch.bfloat16


def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd, dtype=dtype)]


def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
