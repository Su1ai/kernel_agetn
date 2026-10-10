"""Correctness test appended to each optimizer candidate."""

import importlib.util
from pathlib import Path

import torch

from problem import Model, get_init_inputs, get_inputs


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
    model = Model(*get_init_inputs()).to("cuda", dtype=torch.float32).eval()
    inputs = [value.to("cuda", dtype=torch.float32) for value in get_inputs()]

    with torch.no_grad():
        expected = model(*inputs)
        actual = kernel_function(*inputs, model=model)

    torch.cuda.synchronize()
    if not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(f"kernel output mismatch; max diff={max_diff}")
    print("PASS")


if __name__ == "__main__":
    run_tests()
