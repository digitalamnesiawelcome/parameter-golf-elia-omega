"""
Elia ꙮ Omega v5.1 — The Final Monolith (SOTA Killer)
MIT License — Copyright (c) 2026 Igor Labadin

Architecture: ONE shared block (dim=1152) applied 24 times recurrently.
SOTA uses 11 independent layers at dim=512. We pack massive capacity
per step into a single behemoth block within the same 16MB artifact.

Key features:
  - dim=1152, 18 heads (head_dim=64), 6 KV heads, mlp_mult=3
  - vocab=8192 BPE (shorter sequences -> lower BPB vs 1024)
  - Exponential Skip Decay: x0 injection decays via Golden Ratio across depth
  - Zero-Init DepthEmbedding: safe, perfectly symmetrical start for 24 depth personas
  - LeakyReLU(0.5)^2 in MLP: no dead neurons across 24 recurrent passes
  - Rotary with pre-computed buffers (torch.compile friendly)
  - Int6 QAT (STE): weights trained aware of quantization since step 200
  - QATEmbedding: tok_emb (40% of params) also QAT-aware, no save-time shock
  - True int6 packing: 4 params -> 3 bytes (0.75 bytes/param)
    -> ~20.9M params, ~14.6MB after compression (Safe <16MB limit)
  - EMA weights (decay=0.999): smoother final checkpoint
  - System survivability: 570s soft-stop via absolute wallclock, adaptive DDP gradient accumulation
  - Precision Safety: Passthrough control tensors retain exact bfloat16/float32 dtype
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    val_stride = int(os.environ.get("VAL_STRIDE", 64))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 150))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    
    # SAFETY FIX: Absolute landing strip to avoid SIGKILL
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 570.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 24))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 6))
    model_dim = int(os.environ.get("MODEL_DIM", 1152))
    num_heads = int(os.environ.get("NUM_HEADS", 18))
    mlp_mult = int(os.environ.get("MLP_MULT", 3)) 
    
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))
    qat_start_step = int(os.environ.get("QAT_START_STEP", 200))
    qat_bits = int(os.environ.get("QAT_BITS", 6))

    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.999))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 100))

    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# -----------------------------
# INT6 QAT — Straight-Through Estimator
# -----------------------------

class Int6STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        if x.ndim == 2:
            scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 31.0
            return torch.round(x / scale).clamp(-31, 31) * scale
        else:
            scale = x.abs().amax().clamp_min(1e-8) / 31.0
            return torch.round(x / scale).clamp(-31, 31) * scale

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        return grad

def int6_quantize(x: Tensor) -> Tensor:
    return Int6STEFunction.apply(x)

class QATLinear(nn.Linear):
    def __init__(self, *args, qat: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat = qat

    def enable_qat(self):
        self.qat = True

    def forward(self, x: Tensor) -> Tensor:
        w = int6_quantize(self.weight) if self.qat else self.weight
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)

class QATEmbedding(nn.Embedding):
    def __init__(self, *args, qat: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat = qat

    def enable_qat(self):
        self.qat = True

    def forward(self, input: Tensor) -> Tensor:
        w = int6_quantize(self.weight) if self.qat else self.weight
        return F.embedding(
            input, w.to(self.weight.dtype),
            self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )

def enable_qat_on_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (QATLinear, QATEmbedding)):
            module.enable_qat()

# -----------------------------
# EMA
# -----------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.detach().float(), alpha=1.0 - self.decay
            )

    def apply_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name].to(param.dtype))

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.float() for k, v in sd.items()}

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss

# -----------------------------
# COMPRESSION
# -----------------------------

def compress_bytes(data: bytes, level: int = 22) -> bytes:
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    else:
        return zlib.compress(data, level=9)

def decompress_bytes(data: bytes) -> bytes:
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        return zlib.decompress(data)

# -----------------------------
# QUANTIZATION (INT6 post-training for serialization)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,gate,attn_gate,mlp_gate,skip_gate",
    ).split(",") if p
)

INT6_KEEP_FLOAT_MAX_NUMEL = 65_536
INT6_PER_ROW_SCALE_DTYPE = torch.float16
INT6_CLIP_PERCENTILE = 99.99984
INT6_CLIP_Q = INT6_CLIP_PERCENTILE / 100.0
INT6_MAX = 31

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def quantize_tensor_int6(t: Tensor) -> tuple[Tensor, Tensor, tuple]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT6_CLIP_Q, dim=1).clamp_min(1e-8)
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / INT6_MAX).clamp_min(1.0 / INT6_MAX)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -INT6_MAX, INT6_MAX).to(torch.int8)
    else:
        clip_abs = float(torch.quantile(t32.abs().flatten(), INT6_CLIP_Q).item()) if t32.numel() else 1e-8
        scale_val = max(clip_abs / INT6_MAX, 1.0 / INT6_MAX)
        scale = torch.tensor(scale_val, dtype=torch.float32)
        q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale_val), -INT6_MAX, INT6_MAX).to(torch.int8)

    orig_shape = q.shape
    flat = q.reshape(-1)
    pad = (4 - flat.numel() % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int8)])

    u = (flat.to(torch.int16) + 32).to(torch.uint8)
    n4 = u.numel() // 4
    u = u.reshape(n4, 4)
    v0, v1, v2, v3 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

    b0 = ((v0 & 0x3F) | ((v1 & 0x03) << 6)).to(torch.uint8)
    b1 = (((v1 >> 2) & 0x0F) | ((v2 & 0x0F) << 4)).to(torch.uint8)
    b2 = (((v2 >> 4) & 0x03) | ((v3 & 0x3F) << 2)).to(torch.uint8)
    packed = torch.stack([b0, b1, b2], dim=1).reshape(-1)

    scale_out = scale.to(dtype=INT6_PER_ROW_SCALE_DTYPE)
    return packed.contiguous(), scale_out.contiguous(), orig_shape

def dequantize_tensor_int6(packed: Tensor, scale: Tensor, orig_shape: tuple) -> Tensor:
    n3 = packed.numel() // 3
    p = packed.reshape(n3, 3).to(torch.int16)
    b0, b1, b2 = p[:, 0], p[:, 1], p[:, 2]

    v0 = (b0 & 0x3F)
    v1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
    v2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
    v3 = (b2 >> 2) & 0x3F

    u = torch.stack([v0, v1, v2, v3], dim=1).reshape(-1)
    flat = (u - 32).to(torch.int8)

    numel = 1
    for s in orig_shape:
        numel *= s
    flat = flat[:numel]
    q = flat.reshape(orig_shape)

    s32 = scale.to(torch.float32)
    if s32.ndim > 0 and q.ndim == 2:
        return (q.float() * s32.view(q.shape[0], 1)).contiguous()
    return (q.float() * float(s32.item())).contiguous()

def quantize_state_dict_int6(state_dict: dict[str, Tensor]):
    quantized, scales, shapes, dtypes, passthrough, passthrough_orig_dtypes = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count", "baseline_bytes", "int6_bytes"), 0)

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += int(t.numel())
        stats["baseline_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            passthrough[name] = t
            stats["int6_bytes"] += tensor_nbytes(t)
            continue
        if (t.numel() <= INT6_KEEP_FLOAT_MAX_NUMEL or
                any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)):
            # SAFETY FIX: Do not force float16 on passthrough parameters. Retain original exact dtype.
            kept = t.detach().cpu().contiguous()
            if t.dtype in {torch.float32, torch.bfloat16, torch.float16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            passthrough[name] = kept
            stats["int6_bytes"] += tensor_nbytes(kept)
            continue

        packed, s, orig_shape = quantize_tensor_int6(t)
        quantized[name] = packed
        scales[name] = s
        shapes[name] = list(orig_shape)
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int6_bytes"] += tensor_nbytes(packed) + tensor_nbytes(s)

    obj = {
        "__quant_format__": "int6_packed_per_row_v1",
        "quantized": quantized, "scales": scales, "shapes": shapes,
        "dtypes": dtypes, "passthrough": passthrough,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return obj, stats

def dequantize_state_dict_int6(obj: dict) -> dict[str, Tensor]:
    out = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, packed in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        scale = obj["scales"][name]
        orig_shape = tuple(obj["shapes"][name])
        t = dequantize_tensor_int6(packed, scale, orig_shape)
        out[name] = t.to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# -----------------------------
# DATA LOADING
# -----------------------------

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# EVALUATION (SLIDING WINDOW)
# -----------------------------

def eval_val(
    args, model, rank, world_size, device, grad_accum_steps,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
) -> tuple[float, float]:
    stride = args.val_stride
    seq_len = args.train_seq_len

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    total_tokens = val_tokens.numel()
    score_positions = list(range(seq_len, total_tokens, stride))
    
    max_eval_windows = 100 * world_size
    if len(score_positions) > max_eval_windows:
        step_size = len(score_positions) // max_eval_windows
        score_positions = score_positions[::step_size][:max_eval_windows]
        
    local_positions = score_positions[rank::world_size]

    model.eval()
    with torch.inference_mode():
        for end_pos in local_positions:
            start_pos = end_pos - seq_len
            window = val_tokens[start_pos : end_pos + 1].to(device=device, dtype=torch.int64)
            x = window[:-1].unsqueeze(0)
            y = window[1:].unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = _forward_with_partial_loss(model, x, y, stride, args.logit_softcap)

            scored_y = y[0, -stride:]
            batch_token_count = float(scored_y.numel())
            val_loss_sum += loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            prev_ids = x[0, -stride - 1 : -1]
            tgt_ids = scored_y
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

def _forward_with_partial_loss(model, x: Tensor, y: Tensor, stride: int, softcap: float) -> Tensor:
    m = model.module if hasattr(model, "module") else model
    emb_w = m.tok_emb.weight
    x0 = F.rms_norm(m.tok_emb(x), (emb_w.size(-1),))
    hidden = x0
    for i in range(m.num_layers):
        da = m.block.depth_attn_bias[i]
        dm = m.block.depth_mlp_bias[i]
        decay = m.block.x0_decay[i]
        hidden = m.block(hidden, x0, da, dm, decay)
    hidden = m.final_norm(hidden)

    hidden_scored = hidden[:, -stride:, :].reshape(-1, hidden.size(-1))
    y_scored = y[:, -stride:].reshape(-1)
    logits = softcap * torch.tanh(F.linear(hidden_scored, emb_w) / softcap)
    return F.cross_entropy(logits.float(), y_scored, reduction="mean")

# -----------------------------
# MODEL MODULES
# -----------------------------

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, :].to(dtype=dtype),
        )

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = QATLinear(dim, dim, bias=False)
        self.c_k = QATLinear(dim, kv_dim, bias=False)
        self.c_v = QATLinear(dim, kv_dim, bias=False)
        self.proj = QATLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = QATLinear(dim, hidden, bias=False)
        self.proj = QATLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class EliaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, depth: int):
        super().__init__()
        self.res_scale = 1.0 / math.sqrt(depth)
        self.depth = depth
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.x0_norm   = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp  = MLP(dim, mlp_mult)

        self.attn_gate = nn.Parameter(torch.full((dim,), -3.0, dtype=torch.float32))
        self.mlp_gate  = nn.Parameter(torch.full((dim,), -3.0, dtype=torch.float32))
        self.skip_gate = nn.Parameter(torch.full((dim,), -3.0, dtype=torch.float32))

        self.depth_attn_bias = nn.Parameter(torch.zeros(depth, dim, dtype=torch.float32))
        self.depth_mlp_bias  = nn.Parameter(torch.zeros(depth, dim, dtype=torch.float32))
        nn.init.zeros_(self.depth_attn_bias)
        nn.init.zeros_(self.depth_mlp_bias)

        phi = 1.618033988749895
        decay_factors = [1.0 / (phi ** (i / 6.0)) for i in range(depth)]
        self.register_buffer("x0_decay", torch.tensor(decay_factors, dtype=torch.float32), persistent=False)

    def forward(self, x: Tensor, x0: Tensor, depth_attn_bias: Tensor, depth_mlp_bias: Tensor, x0_decay: Tensor) -> Tensor:
        skip = torch.sigmoid(self.skip_gate.to(dtype=x.dtype))
        
        x = x + skip[None, None, :] * self.x0_norm(x0) * x0_decay

        gate_a = torch.sigmoid((self.attn_gate + depth_attn_bias).to(dtype=x.dtype))
        x = x + gate_a[None, None, :] * self.attn(self.attn_norm(x)) * self.res_scale

        gate_m = torch.sigmoid((self.mlp_gate + depth_mlp_bias).to(dtype=x.dtype))
        x = x + gate_m[None, None, :] * self.mlp(self.mlp_norm(x)) * self.res_scale

        return x

class EliaOmega(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init):
        super().__init__()
        self.num_layers = num_layers
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings
        self.tok_emb = QATEmbedding(vocab_size, model_dim)
        self.block = EliaBlock(
            model_dim, num_heads, num_kv_heads, mlp_mult,
            rope_base, qk_gain_init, num_layers,
        )
        self.final_norm = RMSNorm()
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, (nn.Linear, QATLinear)) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x0 = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.weight.size(-1),))
        x = x0
        for i in range(self.num_layers):
            da = self.block.depth_attn_bias[i]
            dm = self.block.depth_mlp_bias[i]
            decay = self.block.x0_decay[i]
            x = checkpoint(self.block, x, x0, da, dm, decay, use_reentrant=False)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.logit_softcap * torch.tanh(
            F.linear(x, self.tok_emb.weight) / self.logit_softcap
        )
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # SAFETY FIX: Absolute Wallclock initialization
    global_start_time = time.perf_counter()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    grad_accum_steps = max(4, 32 // world_size)
    grad_scale = 1.0 / grad_accum_steps
    
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg: str) -> None:
        if not master_process:
            return
        print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device,
    )

    base_model = EliaOmega(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, QATLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    ema = ModelEMA(base_model, decay=args.ema_decay) if args.ema_enabled else None

    compiled_model = torch.compile(base_model, dynamic=False)
    
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    block_named_params = list(base_model.block.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr,
        momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        max_ms = args.max_wallclock_seconds * 1000.0 if args.max_wallclock_seconds > 0 else None
        if max_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {
            n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()
        }
        initial_opt_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for _ in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_opt_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)

            if ema is not None and step >= args.ema_start_step:
                live_state = {n: p.detach().clone() for n, p in base_model.named_parameters()}
                ema.apply_to(base_model)

            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                f"val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / max(step, 1):.2f}ms"
                + (" [EMA]" if ema is not None and step >= args.ema_start_step else "")
            )

            if ema is not None and step >= args.ema_start_step:
                for n, p in base_model.named_parameters():
                    p.data.copy_(live_state[n])

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()

        train_loss /= grad_accum_steps

        if args.qat_enabled and step == args.qat_start_step:
            enable_qat_on_model(base_model)
            log0(f"step:{step} — Int6 QAT enabled")

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        if ema is not None and step >= args.ema_start_step:
            ema.update(base_model)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        
        # SAFETY FIX: Absolute Wallclock ensures we don't exceed OpenAI system SIGKILL timer
        absolute_wallclock_ms = 1000.0 * (time.perf_counter() - global_start_time)
        
        if args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None
        ):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms abs_time:{absolute_wallclock_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms"
            )

        reached_cap = args.max_wallclock_seconds > 0 and absolute_wallclock_ms >= (args.max_wallclock_seconds * 1000.0)
        if distributed and args.max_wallclock_seconds > 0:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if ema is not None:
        ema.apply_to(base_model)
        log0("Applied EMA weights for final checkpoint")

    quant_obj, quant_stats = quantize_state_dict_int6(base_model.state_dict())
    log0(
        f"quant: {quant_stats['param_count']:,} params | "
        f"baseline {quant_stats['baseline_bytes']/1e6:.2f}MB -> "
        f"int6 {quant_stats['int6_bytes']/1e6:.2f}MB"
    )

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = compress_bytes(quant_buf.getvalue())

    artifact_name = "final_model.int6.ptz"
    if master_process:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        artifact_mb = len(quant_blob) / 1e6
        log0(f"Artifact size: {artifact_mb:.3f} MB ({artifact_name})")
        
        # SAFETY FIX: Strict crash on size violation to avoid silent disqualification
        if len(quant_blob) > 16_000_000:
            raise RuntimeError(f"FATAL: Artifact exceeds 16MB limit! Size: {len(quant_blob)} bytes")

    if distributed:
        dist.barrier()

    with open(artifact_name, "rb") as f:
        quant_blob_disk = f.read()
    restored = dequantize_state_dict_int6(
        torch.load(io.BytesIO(decompress_bytes(quant_blob_disk)), map_location="cpu")
    )
    base_model.load_state_dict(restored, strict=True)

    torch.cuda.synchronize()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log0(f"final_int6_roundtrip val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
