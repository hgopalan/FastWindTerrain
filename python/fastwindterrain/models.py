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
           "clip_grad_norm", "d4_average"]

#: Name to builder. Extend here; nothing else needs to know.
ARCHITECTURES = ("fno", "ufno", "unet", "gcnn")


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

    def _rot90(t, r):
        """Rotate the LAST TWO axes of a tensor counter-clockwise."""
        for _ in range(int(r) % 4):
            t = torch.transpose(torch.flip(t, dims=(-2,)), -2, -1)
        return t

    def _act_g(t, g):
        """Apply a D4 element to the spatial axes: mirror in x, then rotate.

        The same convention as training.transform_field and as the
        rotation test that verified the solver at 1e-13. There is exactly
        one convention in this project and this is it.
        """
        r, m = g
        if m:
            t = torch.flip(t, dims=(-1,))
        return _rot90(t, r)

    def _act_vec(u, v, g):
        """The same element on a pair of horizontal components."""
        r, m = g
        uu, vv = _act_g(u, g), _act_g(v, g)
        if m:
            uu = -uu
        th = torch.tensor(r * math.pi / 2.0, dtype=u.dtype, device=u.device)
        c, s = torch.cos(th), torch.sin(th)
        return uu * c - vv * s, uu * s + vv * c

    class GConvD4(nn.Module):
        """Group convolution over D4, features laid out as (B, C*8, H, W).

        The group axis is folded into channels so pooling, interpolation
        and normalisation work unchanged. One weight of shape
        (Cout, Cin, 8, k, k) is expanded at every forward pass into the
        (Cout*8, Cin*8, k, k) bank an ordinary conv2d wants:

            expanded[g][o, (h, i)] = W[o, i, g^-1 h]

        NO SPATIAL TRANSFORM OF THE WEIGHTS, because of the convention the
        lifting layer sets. Lifting by ``conv(g^-1 . x)`` makes a feature
        transform under an input symmetry by PURE PERMUTATION of the group
        axis -- block g becomes block g0^-1 g, with no spatial movement,
        since each block already lives in its own rotated frame. Mixing
        that convention with the textbook one (which does move the
        filters) is what a first attempt did, and it gave 0.48 relative
        equivariance error instead of round-off.

        That is where the saving is -- the
        parameters are shared across the eight group elements rather than
        learned eight times, so a G-CNN of a given width has EIGHT TIMES
        FEWER parameters than a plain CNN of the same width, at comparable
        compute.

        This is strictly more expressive than frame-averaging the output:
        the group axis is carried THROUGH the network, so intermediate
        features are equivariant and layers can mix group elements.
        """

        def __init__(self, cin, cout, k=3, stride=1):
            super().__init__()
            self.cin, self.cout, self.k, self.stride = cin, cout, k, stride
            self.weight = nn.Parameter(
                torch.randn(cout, cin, 8, k, k)
                * (1.0 / math.sqrt(cin * 8 * k * k)))
            self.bias = nn.Parameter(torch.zeros(cout))
            # g^-1 h for every (g, h), precomputed: the group table must
            # not be rebuilt per step, and an index that is wrong by a
            # transpose gives a network that is ALMOST equivariant.
            idx = [[D4_ELEMENTS.index(d4_compose(d4_inverse(g), h))
                    for h in D4_ELEMENTS] for g in D4_ELEMENTS]
            self.register_buffer("gidx", torch.tensor(idx), persistent=False)

        def forward(self, x):
            blocks = []
            for gi, g in enumerate(D4_ELEMENTS):
                # (cout, cin, 8, k, k) reordered on the group axis, then
                # the kernel itself moved by g.
                # (cout, cin, 8, k, k) -> (cout, 8, cin, k, k) BEFORE the
                # reshape. The feature tensor is laid out group-major
                # (block g is channels g*C .. g*C+C), so the weight has to
                # be too. Reshaping without this permute pairs weight
                # (i, h) against input (h, i) and gives a network that is
                # 0.16 relative off equivariance -- close enough to look
                # like a subtle group-theory error and not be one.
                w = self.weight[:, :, self.gidx[gi]]
                w = w.permute(0, 2, 1, 3, 4).reshape(
                    self.cout, 8 * self.cin, self.k, self.k)
                blocks.append(w)
            W = torch.cat(blocks, dim=0)
            b = self.bias.repeat_interleave(1).repeat(8)
            return F.conv2d(x, W, b, stride=self.stride,
                            padding=self.k // 2)

    def _gcat(a, b):
        """Concatenate two group-major tensors BLOCK BY BLOCK.

        Both are laid out (B, C*8, H, W) with block g occupying channels
        g*C .. g*C+C. A plain torch.cat appends all of one then all of the
        other, which produces [a_g0..a_g7, b_g0..b_g7] and destroys the
        blocking -- the next group convolution then reads channel h of
        block g from the wrong tensor. It is invisible except as an
        equivariance error, which is exactly how it was found.
        """
        ca, cb = a.shape[1] // 8, b.shape[1] // 8
        return torch.cat(
            [torch.cat([a[:, g * ca:(g + 1) * ca],
                        b[:, g * cb:(g + 1) * cb]], dim=1)
             for g in range(8)], dim=1)

    class GCNN(nn.Module):
        """A U-Net whose convolutions are group convolutions over D4.

        Equivariance is a property of the weights rather than something
        taught by augmentation or bolted on at inference. The lifting and
        projection layers are where the channel semantics live: terrain
        and slope move, the direction planes rotate as a vector, and on
        the way out (u, v) per level rotates while w does not.
        """

        def __init__(self, in_ch, out_ch, width=12, depth=4, n_levels=9,
                     n_scalar_in=2):
            super().__init__()
            self.n_levels, self.ns = n_levels, n_scalar_in
            self.in_ch = in_ch
            chs = [width * 2 ** min(i, 2) for i in range(depth)]
            self.lift = nn.Conv2d(in_ch, chs[0], 3, padding=1)
            self.down = nn.ModuleList()
            prev = chs[0]
            for c in chs:
                self.down.append(nn.ModuleList(
                    [GConvD4(prev, c), GConvD4(c, c)]))
                prev = c
            self.mid = GConvD4(prev, prev)
            self.up = nn.ModuleList([
                GConvD4(chs[i] + (chs[i + 1] if i + 1 < depth else prev),
                        chs[i]) for i in range(depth)])
            # Acts on ONE group block, shared across all eight.
            self.out = nn.Conv2d(chs[0], out_ch, 1)

        def _lift(self, x):
            """(B, Cin, H, W) -> (B, C*8, H, W), one block per element.

            Built by transforming the INPUT eight ways and sharing one
            ordinary convolution, which is the same operator as
            transforming the filter and avoids having to encode the
            vector semantics inside a weight expansion.
            """
            ns, out = self.ns, []
            for g in D4_ELEMENTS:
                gi = d4_inverse(g)
                parts = [_act_g(x[:, :ns], gi)]
                u, v = _act_vec(x[:, ns:ns + 1], x[:, ns + 1:ns + 2], gi)
                parts += [u, v]
                if x.shape[1] > ns + 2:
                    parts.append(_act_g(x[:, ns + 2:], gi))
                out.append(self.lift(torch.cat(parts, dim=1)))
            return torch.cat(out, dim=1)

        def _project(self, feat):
            """(B, C*8, H, W) -> (B, out, H, W), undoing each element."""
            n, c = self.n_levels, feat.shape[1] // 8
            acc = None
            for gi, g in enumerate(D4_ELEMENTS):
                y = self.out(feat[:, gi * c:(gi + 1) * c])
                u, v = _act_vec(y[:, :n], y[:, n:2 * n], g)
                w = _act_g(y[:, 2 * n:], g)
                z = torch.cat([u, v, w], dim=1)
                acc = z if acc is None else acc + z
            return acc / 8.0

        def forward(self, x):
            h = self._lift(x)
            skips, sizes = [], []
            for a, b in self.down:
                h = F.gelu(b(F.gelu(a(h))))
                skips.append(h)
                sizes.append(h.shape[-2:])
                h = F.avg_pool2d(h, 2, ceil_mode=True)
            h = F.gelu(self.mid(h))
            for i in range(len(self.up) - 1, -1, -1):
                h = F.interpolate(h, size=sizes[i], mode="bilinear",
                                  align_corners=False)
                h = F.gelu(self.up[i](_gcat(skips[i], h)))
            return self._project(h)

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

    return SpectralNet, UNet, GCNN


def build(name, in_channels, out_channels, **kw):
    """Construct an architecture by name.

    ``kw`` passes through to the class, so a sweep over width or modes is
    a dict rather than an edit here.
    """
    name = str(name).lower()
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; "
                         f"expected one of {ARCHITECTURES}")
    SpectralNet, UNet, GCNN = _modules()
    if name == "unet":
        return UNet(in_channels, out_channels, **kw)
    if name == "gcnn":
        return GCNN(in_channels, out_channels, **kw)
    if name == "fno":
        kw.setdefault("unet_blocks", 0)
    else:                                   # ufno
        # Wen et al. put the U-Net path in the later layers only: early
        # layers do the global work the spectrum is good at, and the local
        # correction is applied once the representation is formed.
        kw.setdefault("unet_blocks", 2)
    return SpectralNet(in_channels, out_channels, **kw)


def d4_average(model, n_levels=9, n_scalar_in=2):
    """Wrap a model so it is EXACTLY equivariant under the square's group.

    Frame averaging: run the model on all eight symmetries of the input,
    map each output back, and average. For a finite group this makes any
    network exactly equivariant with no architectural change --

        f(x) = (1/|G|) sum_g  g^-1 . model(g . x)

    -- and, unlike augmentation, the guarantee holds for weights that were
    never trained for it. That is the point: the learning curve showed D4
    augmentation still buying 7 % at the plateau, so the model never fully
    learns the symmetry from data even with the whole corpus. This closes
    that gap by construction.

    It costs eight forward passes. The cheaper form is a group-equivariant
    convolution, which ties the weights instead of averaging the outputs;
    this exists first because it can be measured on an ALREADY TRAINED
    model, which sizes the prize before anyone pays for the architecture.

    CHANNEL SEMANTICS ARE NOT OPTIONAL. Under a rotation the terrain and
    slope planes merely move, the direction planes rotate as a vector, and
    the same split applies to the output: (u, v) per level is a vector, w
    is a scalar. Treating a vector as a scalar produces a field that looks
    right and points the wrong way, which no loss curve would reveal.
    ``n_scalar_in`` is how many leading input channels are scalars (two:
    terrain and slope); the next two are the direction vector, and
    anything after them is scalar again (the spectral descriptors).
    """
    import torch
    import torch.nn as nn

    from .training import D4_OPS

    def _spatial(t, ang, mir):
        if mir:
            t = torch.flip(t, dims=(-1,))
        for _ in range(int(round(ang)) % 360 // 90):
            t = torch.transpose(torch.flip(t, dims=(-2,)), -2, -1)
        return t

    def _vector(a, b, ang, mir):
        aa, bb = _spatial(a, ang, mir), _spatial(b, ang, mir)
        if mir:
            aa = -aa
        th = torch.tensor(ang * torch.pi / 180.0, dtype=a.dtype,
                          device=a.device)
        c, s = torch.cos(th), torch.sin(th)
        return aa * c - bb * s, aa * s + bb * c

    class _D4Average(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            ns, out = n_scalar_in, None
            for ang, mir in D4_OPS:
                parts = [_spatial(x[:, :ns], ang, mir)]
                ux, uy = _vector(x[:, ns:ns + 1], x[:, ns + 1:ns + 2],
                                 ang, mir)
                parts += [ux, uy]
                if x.shape[1] > ns + 2:
                    parts.append(_spatial(x[:, ns + 2:], ang, mir))
                y = self.inner(torch.cat(parts, dim=1))

                # Map the output BACK. The forward transform is mirror
                # then rotate, so its inverse is rotate by -ang then
                # mirror -- getting the order wrong is silent.
                n = n_levels
                u, v = y[:, :n], y[:, n:2 * n]
                w = y[:, 2 * n:]
                u, v = _vector(u, v, -ang, False)
                w = _spatial(w, -ang, False)
                if mir:
                    u, v = _spatial(u, 0, True), _spatial(v, 0, True)
                    u = -u
                    w = _spatial(w, 0, True)
                z = torch.cat([u, v, w], dim=1)
                out = z if out is None else out + z
            return out / float(len(D4_OPS))

    return _D4Average(model)


#: The D4 group as ``(rotation index 0-3, mirror 0/1)``, in the same order
#: as :data:`fastwindterrain.training.D4_OPS`. An element means "mirror in
#: x if m, then rotate by 90r degrees counter-clockwise", which is the
#: convention the whole project uses and the one verified against the
#: solver at 1e-13.
D4_ELEMENTS = ((0, 0), (1, 0), (2, 0), (3, 0),
               (0, 1), (1, 1), (2, 1), (3, 1))


def d4_compose(g1, g2):
    """``g1 . g2``: apply g2 first, then g1.

    With g = R^r M^m and the relation ``M R = R^-1 M``, the product is
    ``R^(r1 + r2 (-1)^m1) M^(m1 + m2)``. Derived once here because a
    group table that is wrong by a transpose gives a network that is
    almost equivariant, which is worse than one that is obviously not.
    """
    r1, m1 = g1
    r2, m2 = g2
    return ((r1 + (r2 if m1 == 0 else -r2)) % 4, (m1 + m2) % 2)


def d4_inverse(g):
    r, m = g
    return ((-r) % 4 if m == 0 else r, m)
