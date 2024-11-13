# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

class GradMultiply(torch.autograd.Function):
    """
    Example call GradMultiply.apply(x, 2.0) to multiply the gradient of x by 2.0.
    """
    @staticmethod
    def forward(ctx, x, scale) -> torch.Tensor:
        ctx.scale = scale
        res = x.new(x)
        return res
    
    @staticmethod
    def backward(ctx, grad) -> tuple[torch.Tensor, None]:
        return grad * ctx.scale, None
    