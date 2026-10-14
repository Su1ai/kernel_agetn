# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Platform configuration registry for multi-backend support.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend."""

    name: str
    device_string: str
    guidance_block: str
    kernel_guidance: str
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)


# Platform-specific constants
_XPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR INTEL XPU:**
- Default tensor allocations to device='xpu' (never 'cuda'); CPU is allowed only when necessary.
- Check availability with: hasattr(torch, 'xpu') and torch.xpu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device
- Do NOT set TRITON_BACKENDS environment variable
- Do NOT import or disable XPUDriver
- Use torch.xpu.synchronize() if synchronization is needed
- Intel XPU subgroup size is typically 16 (not 32 like CUDA warps)
- Preferred block sizes: 64, 128, 256, or 512"""

_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

_XPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
    "from triton.backends.intel.driver import XPUDriver",
    "XPUDriver.is_available = classmethod(lambda cls: False)",
)


_PPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR PPU:**
- PPU uses a CUDA-compatible PyTorch/Triton runtime; keep device='cuda' and torch.cuda synchronization calls.
- CUDA compatibility does NOT mean the device is an NVIDIA GPU or an A100.
- Do not use NVIDIA-only APIs such as tl.extra.cuda experimental TMA helpers.
- Do not assume Hopper/Blackwell TMA, warp specialization, NVIDIA FP8 formats, SM limits, or Tensor Core behavior.
- Use only portable Triton operations unless a capability is explicitly reported as supported."""

_PPU_KERNEL_GUIDANCE = """\
## PPU-Specific Generation Constraints

The target is a PPU accelerator exposed through a CUDA-compatible runtime.
Generate portable Triton code using tl.load, tl.store, tl.dot, reductions, masking,
and ordinary num_warps/num_stages tuning. Do not infer NVIDIA hardware features
from the CUDA-compatible device string or reported compatibility capability.
Avoid NVIDIA Hopper/Blackwell-only TMA, warp-specialization, tl.extra.cuda, and
NVIDIA-specific FP8 techniques unless the supplied capability profile explicitly
enables them."""

# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
    "ppu": PlatformConfig(
        name="ppu",
        device_string="cuda",
        guidance_block=_PPU_GUIDANCE,
        kernel_guidance=_PPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=(),
    ),
}


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
