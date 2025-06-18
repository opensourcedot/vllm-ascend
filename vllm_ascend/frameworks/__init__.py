import sys
import torch
import torch_npu

from vllm.frameworks import current_framework

npu = None

if current_framework is None:
    raise RuntimeError(
        "current_framework is not set. Please set the VLLM_FRAMEWORK environment variable to the desired framework."
    )
elif current_framework == torch:
    npu = torch_npu
else:
    # If the current framework is not PyTorch, we assume it is a custom framework
    # that has its own NPU module.
    npu = current_framework.npu

__all__ = ["npu"]