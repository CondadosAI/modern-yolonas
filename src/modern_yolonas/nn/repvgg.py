"""QARepVGGBlock — quantization-aware RepVGG block.

Attribute names mirror super-gradients exactly for state_dict compatibility:
  branch_3x3  (Sequential: conv + bn)
  branch_1x1  (Conv2d with bias)
  identity     (Residual or None)
  post_bn      (BatchNorm2d or Identity)
  nonlinearity (activation)
  se           (squeeze-excite or Identity)
  rbr_reparam  (fused Conv2d)
  alpha        (Parameter or 1.0)
  id_tensor    (buffer)
"""

from __future__ import annotations

from typing import Type, Mapping, Any

import torch
from torch import nn


class Residual(nn.Module):
    """Identity skip connection."""

    def forward(self, x):
        return x


class QARepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation_type: Type[nn.Module] = nn.ReLU,
        activation_kwargs: Mapping[str, Any] | None = None,
        se_type: Type[nn.Module] = nn.Identity,
        se_kwargs: Mapping[str, Any] | None = None,
        build_residual_branches: bool = True,
        use_residual_connection: bool = True,
        use_alpha: bool = False,
        use_1x1_bias: bool = True,
        use_post_bn: bool = True,
    ):
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {}
        if se_kwargs is None:
            se_kwargs = {}

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activation_type = activation_type
        self.activation_kwargs = activation_kwargs
        self.se_type = se_type
        self.se_kwargs = se_kwargs
        self.use_residual_connection = use_residual_connection
        self.use_alpha = use_alpha
        self.use_1x1_bias = use_1x1_bias
        self.use_post_bn = use_post_bn

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

        # 3x3 branch: conv + bn
        self.branch_3x3 = nn.Sequential()
        self.branch_3x3.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation,
            ),
        )
        self.branch_3x3.add_module("bn", nn.BatchNorm2d(out_channels))

        # 1x1 branch: conv with bias
        self.branch_1x1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
            bias=use_1x1_bias,
        )

        # Identity branch
        if use_residual_connection:
            assert out_channels == in_channels and stride == 1
            self.identity = Residual()

            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3))
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 1, 1] = 1.0

            self.register_buffer(
                "id_tensor",
                id_tensor.to(dtype=self.branch_1x1.weight.dtype),
                persistent=False,
            )
        else:
            self.identity = None

        # Alpha weighting for 1x1 branch
        if use_alpha:
            noise = torch.randn((1,)) * 0.01
            self.alpha = nn.Parameter(torch.tensor([1.0]) + noise, requires_grad=True)
        else:
            self.alpha = 1.0

        # Post-BN
        if self.use_post_bn:
            self.post_bn = nn.BatchNorm2d(out_channels)
        else:
            self.post_bn = nn.Identity()

        # Fused reparam conv (used after partial/full fusion)
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True,
        )

        self.partially_fused = False
        self.fully_fused = False

        if not build_residual_branches:
            self.fuse_block_residual_branches()

    def forward(self, inputs):
        if self.fully_fused:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.partially_fused:
            return self.se(self.nonlinearity(self.post_bn(self.rbr_reparam(inputs))))

        if self.identity is None:
            id_out = 0.0
        else:
            id_out = self.identity(inputs)

        x_3x3 = self.branch_3x3(inputs)
        x_1x1 = self.alpha * self.branch_1x1(inputs)

        branches = x_3x3 + x_1x1 + id_out
        out = self.nonlinearity(self.post_bn(branches))
        return self.se(out)

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    def _get_equivalent_kernel_bias_for_branches(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(
            self.branch_3x3.conv.weight,
            0,
            self.branch_3x3.bn.running_mean,
            self.branch_3x3.bn.running_var,
            self.branch_3x3.bn.weight,
            self.branch_3x3.bn.bias,
            self.branch_3x3.bn.eps,
        )

        kernel1x1 = self._pad_1x1_to_3x3_tensor(self.branch_1x1.weight)
        bias1x1 = self.branch_1x1.bias if self.branch_1x1.bias is not None else 0

        kernelid = self.id_tensor if self.identity is not None else 0
        biasid = 0

        eq_kernel = kernel3x3 + self.alpha * kernel1x1 + kernelid
        eq_bias = bias3x3 + self.alpha * bias1x1 + biasid
        return eq_kernel, eq_bias

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(kernel, bias, running_mean, running_var, gamma, beta, eps):
        std = torch.sqrt(running_var + eps)
        b = beta - gamma * running_mean / std
        A = gamma / std
        A_ = A.expand_as(kernel.transpose(0, -1)).transpose(0, -1)
        return kernel * A_, bias * A + b

    def partial_fusion(self):
        if self.partially_fused:
            return
        if self.fully_fused:
            raise NotImplementedError(
                "QARepVGGBlock can't be converted to partially fused from fully fused"
            )
        kernel, bias = self._get_equivalent_kernel_bias_for_branches()
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        if hasattr(self, "alpha") and isinstance(self.alpha, nn.Parameter):
            self.__delattr__("alpha")
        elif hasattr(self, "alpha"):
            delattr(self, "alpha")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.partially_fused = True
        self.fully_fused = False

    def full_fusion(self):
        """Fuse all branches including post_bn into a single conv.

        This is an inference-only operation — all parameters are detached
        from the computation graph afterward. Do not call during training.
        """
        if self.fully_fused:
            return
        if self.training:
            raise RuntimeError(
                "full_fusion() disables gradient tracking and must not be called "
                "during training. Call model.eval() first."
            )
        if not self.partially_fused:
            self.partial_fusion()
        if self.use_post_bn:
            eq_kernel, eq_bias = self._fuse_bn_tensor(
                self.rbr_reparam.weight,
                self.rbr_reparam.bias,
                self.post_bn.running_mean,
                self.post_bn.running_var,
                self.post_bn.weight,
                self.post_bn.bias,
                self.post_bn.eps,
            )
            self.rbr_reparam.weight.data = eq_kernel
            self.rbr_reparam.bias.data = eq_bias
        for para in self.parameters():
            para.detach_()
        if hasattr(self, "post_bn"):
            self.__delattr__("post_bn")
        self.partially_fused = False
        self.fully_fused = True

    def fuse_block_residual_branches(self):
        self.partial_fusion()
