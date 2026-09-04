"""
models -- the surrogate architectures, behind one interface.

Phase 22b. Three of them, selected by name, all with the same signature:

    fno     the plain Fourier neural operator (Li et al. 2021)
    ufno    U-FNO: a U-Net branch alongside the spectral one in the later
            layers (Wen et al. 2022). The paper's architecture.
    unet    a convolutional baseline with no spectral path at all

ARCHITECTURE IS A FLAG BECAUSE THE COMPARISON IS THE POINT. The first
question anyone asks of an FNO paper is whether a U-Net does as well, and
that question is cheap to answer if the alternative is one string and
expensive if it means a second codebase. Nothing here is tuned; the
defaults are the published ones, and any claim that one architecture beats
another has to come from a run, not from this file.

PADDING, BECAUSE TERRAIN IS NOT PERIODIC. The FFT treats the domain as
periodic, so the ridge on the west edge is a neighbour of the valley on
the east. That is false for a 5 km window cut out of a landscape, and the
wrap-around shows up as error along the boundaries. Every spectral model
here pads the field before the transform and crops afterwards, which is
the standard remedy and is not optional -- ``padding=0`` is available so
the effect can be measured, not because it is a reasonable default.
"""

import math

__all__ = ["ARCHITECTURES", "build", "count_parameters",
           "clip_grad_norm"]

#: Name to builder. Extend here; nothing else needs to know.
ARCHITECTURES = ("fno", "ufno", "unet")


def _torch():
    import torch
    return torch


def clip_grad_norm(parameters, max_norm):
    """Gradient-norm clipping that survives complex parameters.

    ``torch.nn.utils.clip_grad_norm_`` raises "norm ops are not supported
    for complex yet" on the spectral weights, which are ``cfloat`` by
    construction. Viewing a complex gradient as its real pair gives the
    same sum of squares, so the total norm and the scaling are identical
    to torch's -- this is a workaround for a missing kernel, not a
    different algorithm.

    Clipping is not optional here: the first training run died on a
    single loss spike at epoch 11 and never recovered.
    """
    import torch

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum(
        (torch.view_as_real(g) if g.is_complex() else g).pow(2).sum()
        for g in grads))
    scale = float(max_norm) / (float(total) + 1e-6)
    if scale < 1.0:
        for g in grads:
            g.mul_(scale)
    return float(total)


def count_parameters(model):
    """Trainable parameter count -- the honest size of a model, and the
    number a cost comparison between architectures rests on."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _modules():
    """Build the classes lazily.

    torch is an optional dependency (the ``train`` extra), so importing
    this module must not require it -- ``ARCHITECTURES`` and the
    docstrings stay readable without a 2 GB install.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SpectralConv2d(nn.Module):
        """Multiply the lowest Fourier modes by a learned complex tensor.

        The whole idea of the operator: a global convolution at the cost
        of a truncated spectrum, so the receptive field is the domain
        after one layer rather than after many.
        """

        def __init__(self, cin, cout, modes1, modes2):
            super().__init__()
            self.cin, self.cout = cin, cout
            self.modes1, self.modes2 = modes1, modes2
            scale = 1.0 / (cin * cout)
            # Two blocks: the positive and negative first-axis modes. rfft2
            # keeps the whole first axis and half the second, so the low
            # modes live at both ends of axis one.
            self.w1 = nn.Parameter(scale * torch.randn(
                cin, cout, modes1, modes2, dtype=torch.cfloat))
            self.w2 = nn.Parameter(scale * torch.randn(
                cin, cout, modes1, modes2, dtype=torch.cfloat))

        def forward(self, x):
            b, _, h, w = x.shape
            m1 = min(self.modes1, h // 2)
            m2 = min(self.modes2, w // 2 + 1)
            xf = torch.fft.rfft2(x, norm="ortho")
            out = torch.zeros(b, self.cout, h, w // 2 + 1,
                              dtype=torch.cfloat, device=x.device)
            out[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", xf[:, :, :m1, :m2],
                self.w1[:, :, :m1, :m2])
            out[:, :, -m1:, :m2] = torch.einsum(
                "bixy,ioxy->boxy", xf[:, :, -m1:, :m2],
                self.w2[:, :, :m1, :m2])
            return torch.fft.irfft2(out, s=(h, w), norm="ortho")

    class UNetBranch(nn.Module):
        """A small two-level U-Net, the 'U' in U-FNO.

        It exists to carry what the truncated spectrum cannot: the sharp,
        local structure at the terrain surface, which is exactly where the
        error was measured to live. Odd sizes are handled by interpolating
        back to the input shape rather than assuming a power of two -- the
        corpus grid is 100 x 100.
        """

        def __init__(self, ch):
            super().__init__()
            self.d1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
            self.d2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
            self.u2 = nn.Conv2d(ch, ch, 3, padding=1)
            self.u1 = nn.Conv2d(2 * ch, ch, 3, padding=1)
            self.out = nn.Conv2d(2 * ch, ch, 3, padding=1)

        def forward(self, x):
            s = x.shape[-2:]
            a = F.gelu(self.d1(x))
            b = F.gelu(self.d2(a))
            b = F.interpolate(F.gelu(self.u2(b)), size=a.shape[-2:],
                              mode="bilinear", align_corners=False)
            a = F.gelu(self.u1(torch.cat([a, b], dim=1)))
            a = F.interpolate(a, size=s, mode="bilinear",
                              align_corners=False)
            return self.out(torch.cat([x, a], dim=1))

    class FourierBlock(nn.Module):
        """Spectral path + pointwise path, and optionally the U-Net path.

        With ``unet=True`` this is Wen et al.'s U-Fourier layer; without
        it, Li et al.'s Fourier layer. One class, so the only difference
        between the two architectures in this file is where that flag is
        set.
        """

        def __init__(self, ch, modes1, modes2, unet=False, act=True,
                     norm=True, groups=8):
            super().__init__()
            self.spectral = SpectralConv2d(ch, ch, modes1, modes2)
            self.pointwise = nn.Conv2d(ch, ch, 1)
            self.unet = UNetBranch(ch) if unet else None
            self.act = act
            # Without this the block output is unbounded and drifts until
            # one batch produces a large activation, a gradient a thousand
            # times the usual, and a model that never recovers. Measured:
            # a bare version spiked at step 557 with a gradient norm of
            # 390 against a typical 0.3, and sat at the zero solution for
            # the next 850 steps. Gradient clipping alone did not save it.
            self.norm = (nn.GroupNorm(min(groups, ch), ch) if norm
                         else nn.Identity())

        def forward(self, x):
            y = self.spectral(x) + self.pointwise(x)
            if self.unet is not None:
                y = y + self.unet(x)
            y = self.norm(y)
            return F.gelu(y) if self.act else y

    class SpectralNet(nn.Module):
        """FNO and U-FNO differ only in how many blocks carry a U-Net."""

        def __init__(self, in_ch, out_ch, width=32, modes=16, blocks=4,
                     unet_blocks=0, padding=8, norm=True):
            super().__init__()
            self.padding = int(padding)
            self.lift = nn.Conv2d(in_ch, width, 1)
            self.blocks = nn.ModuleList([
                FourierBlock(width, modes, modes,
                             unet=(i >= blocks - unet_blocks),
                             act=(i < blocks - 1), norm=norm)
                for i in range(blocks)])
            self.project = nn.Sequential(
                nn.Conv2d(width, 2 * width, 1), nn.GELU(),
                nn.Conv2d(2 * width, out_ch, 1))

        def forward(self, x):
            p = self.padding
            if p:
                # Reflect, not zero: a zero border is a cliff to the
                # spectrum and rings worse than the wrap-around it is
                # there to prevent.
                x = F.pad(x, (p, p, p, p), mode="reflect")
            x = self.lift(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.project(x)
            return x[..., p:x.shape[-2] - p, p:x.shape[-1] - p] if p else x

    class UNet(nn.Module):
        """The baseline: no spectral path anywhere.

        Depth is chosen so its receptive field is comparable and its
        parameter count is in the same order -- an unfairly small baseline
        proves nothing, and it is the comparison a reviewer will make
        first.
        """

        def __init__(self, in_ch, out_ch, width=32, depth=4):
            super().__init__()

            def blk(a, b):
                return nn.Sequential(
                    nn.Conv2d(a, b, 3, padding=1), nn.GELU(),
                    nn.Conv2d(b, b, 3, padding=1), nn.GELU())

            chs = [width * 2 ** min(i, 3) for i in range(depth)]
            self.down = nn.ModuleList()
            prev = in_ch
            for c in chs:
                self.down.append(blk(prev, c))
                prev = c
            self.mid = blk(prev, prev)
            self.up = nn.ModuleList(
                [blk(chs[i] + (chs[i + 1] if i + 1 < depth else prev),
                     chs[i]) for i in range(depth)])
            self.out = nn.Conv2d(chs[0], out_ch, 1)

        def forward(self, x):
            skips, sizes = [], []
            for d in self.down:
                x = d(x)
                skips.append(x)
                sizes.append(x.shape[-2:])
                x = F.avg_pool2d(x, 2, ceil_mode=True)
            x = self.mid(x)
            for i in range(len(self.up) - 1, -1, -1):
                x = F.interpolate(x, size=sizes[i], mode="bilinear",
                                  align_corners=False)
                x = self.up[i](torch.cat([skips[i], x], dim=1))
            return self.out(x)

    return SpectralNet, UNet


def build(name, in_channels, out_channels, **kw):
    """Construct an architecture by name.

    ``kw`` passes through to the class, so a sweep over width or modes is
    a dict rather than an edit here.
    """
    name = str(name).lower()
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; "
                         f"expected one of {ARCHITECTURES}")
    SpectralNet, UNet = _modules()
    if name == "unet":
        return UNet(in_channels, out_channels, **kw)
    if name == "fno":
        kw.setdefault("unet_blocks", 0)
    else:                                   # ufno
        # Wen et al. put the U-Net path in the later layers only: early
        # layers do the global work the spectrum is good at, and the local
        # correction is applied once the representation is formed.
        kw.setdefault("unet_blocks", 2)
    return SpectralNet(in_channels, out_channels, **kw)
