"""Correctness test appended to each optimizer candidate."""

import importlib.util
from pathlib import Path

import torch

from problem import Model, get_init_inputs, get_inputs

# problem.py owns the dtype; keeping one source of truth means the PyTorch
# reference and the Triton kernel are compared at equal precision.
try:
    from problem import dtype as _DTYPE
except ImportError:  # older problem.py files predate the dtype export
    _DTYPE = torch.float32


_kernel_spec = importlib.util.spec_from_file_location(
    "kernel", Path(__file__).with_name("kernel.py")
)
if _kernel_spec is None or _kernel_spec.loader is None:
    raise ImportError("Unable to load kernel.py")
_kernel_module = importlib.util.module_from_spec(_kernel_spec)
_kernel_spec.loader.exec_module(_kernel_module)
kernel_function = _kernel_module.kernel_function


def run_tests():
    torch.manual_seed(0)
    model = Model(*get_init_inputs()).to("cuda", dtype=_DTYPE).eval()
    inputs = [value.to("cuda", dtype=_DTYPE) for value in get_inputs()]

    with torch.no_grad():
        expected = model(*inputs)
        actual = kernel_function(*inputs, model=model)

    torch.cuda.synchronize()
    # The reference and the kernel now share one dtype, so the tolerance is set
    # for that precision rather than for fp32: bf16 carries ~3 decimal digits.
    rtol, atol = (1e-2, 1e-2)
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(f"kernel output mismatch; max diff={max_diff}")
    print("PASS")


if __name__ == "__main__":
    run_tests()
