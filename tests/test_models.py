"""
The surrogate architectures (fastwindterrain.models), phase 22b.

These are shape-and-contract tests, not accuracy tests: whether U-FNO
beats a U-Net is a question for a training run, and asserting it here
would be asserting a result rather than testing code. What is worth
pinning is that all three architectures are interchangeable, that the
spectral models really do treat the domain as non-periodic, and that the
registry refuses a name it does not know instead of silently building
something else.
"""

import pytest

torch = pytest.importorskip("torch")

from fastwindterrain import models as M           # noqa: E402

IN, OUT, N = 4, 27, 32          # the real pipeline's channel counts


@pytest.mark.parametrize("arch", M.ARCHITECTURES)
def test_every_architecture_is_interchangeable(arch):
    """Same signature, same input, same output shape. This is what makes
    the architecture a flag rather than a fork of the codebase."""
    m = M.build(arch, IN, OUT)
    y = m(torch.randn(2, IN, N, N))
    assert y.shape == (2, OUT, N, N)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("arch", M.ARCHITECTURES)
def test_every_architecture_produces_gradients(arch):
    """A model whose gradient does not reach its parameters trains to
    nothing while the loop reports a falling loss from the rest."""
    m = M.build(arch, IN, OUT)
    m(torch.randn(1, IN, N, N)).pow(2).mean().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or not p.grad.any())]
    assert not dead, f"no gradient reached: {dead[:5]}"


@pytest.mark.parametrize("arch", M.ARCHITECTURES)
def test_every_architecture_handles_the_corpus_grid(arch):
    """100 x 100 is not a power of two, and the U-Net paths downsample.
    An architecture that silently returns 96 x 96 would misalign every
    field against its terrain."""
    m = M.build(arch, IN, OUT)
    assert m(torch.randn(1, IN, 100, 100)).shape == (1, OUT, 100, 100)


def test_the_spectral_models_are_not_periodic_by_default():
    """The FFT treats the domain as periodic, so without padding the west
    edge is a neighbour of the east one. That is false for a 5 km window
    cut from a landscape, and the padding is what prevents it.
    """
    m = M.build("fno", IN, OUT)
    assert m.padding > 0, "spectral models must pad by default"
    # And it is a real choice, not a constant: padding=0 must still work,
    # so the effect can be measured rather than argued about.
    assert M.build("fno", IN, OUT, padding=0)(
        torch.randn(1, IN, N, N)).shape == (1, OUT, N, N)


def test_ufno_carries_a_unet_branch_and_fno_does_not():
    """The only structural difference between the two, and the thing the
    paper's contribution rests on."""
    fno = M.build("fno", IN, OUT)
    ufno = M.build("ufno", IN, OUT)
    assert sum(b.unet is not None for b in fno.blocks) == 0
    assert sum(b.unet is not None for b in ufno.blocks) == 2
    assert M.count_parameters(ufno) > M.count_parameters(fno)


def test_the_unet_baseline_is_not_unfairly_small():
    """A baseline starved of capacity proves nothing, and it is the first
    thing a reviewer checks. It must be at least the same order as the
    model it is a baseline for."""
    ufno = M.count_parameters(M.build("ufno", IN, OUT))
    unet = M.count_parameters(M.build("unet", IN, OUT))
    assert unet > 0.5 * ufno, f"unet {unet:,} against ufno {ufno:,}"


def test_the_spectral_blocks_are_normalised_by_default():
    """The defect that cost two training runs. Without normalisation the
    block output is unbounded: a bare U-FNO spiked at step 557 with a
    gradient norm of 390 against a typical 0.3, and then sat at the zero
    solution -- loss exactly 1.0 on unit-variance targets -- for the next
    850 steps. Gradient clipping did not save it, because by then the
    model was already in a dead region.

    With GroupNorm, over the same 1400 steps: worst step loss 1.17
    against 23.30, worst gradient 2.5 against 390, final loss 0.29
    against 1.00.
    """
    torch_nn = pytest.importorskip("torch").nn
    for arch in ("fno", "ufno"):
        m = M.build(arch, IN, OUT)
        assert all(isinstance(b.norm, torch_nn.GroupNorm)
                   for b in m.blocks), arch
    # Available to turn off, so the effect stays measurable.
    off = M.build("fno", IN, OUT, norm=False)
    assert all(isinstance(b.norm, torch_nn.Identity) for b in off.blocks)


def test_gradient_clipping_survives_complex_parameters():
    """torch's clip_grad_norm_ raises on the cfloat spectral weights, so
    the models module carries its own. It must agree with torch's on a
    real-valued model, or it is a different algorithm rather than a
    workaround for a missing kernel."""
    m = M.build("unet", IN, OUT)          # real parameters only
    m(torch.randn(1, IN, N, N)).pow(2).mean().backward()
    mine = M.clip_grad_norm(m.parameters(), 1e9)     # measure, do not clip
    theirs = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9))
    assert mine == pytest.approx(theirs, rel=1e-5)

    # It clips when the norm is above the limit, and leaves it alone when
    # it is below -- the second half matters as much as the first, since a
    # clip that always fires is a learning-rate change in disguise.
    target = mine / 2.0
    M.clip_grad_norm(m.parameters(), target)
    assert M.clip_grad_norm(m.parameters(), 1e9) == pytest.approx(target,
                                                                  rel=1e-4)
    M.clip_grad_norm(m.parameters(), 1e9)
    assert M.clip_grad_norm(m.parameters(), 1e9) == pytest.approx(target,
                                                                  rel=1e-4)


def test_clipping_a_complex_model_does_not_raise():
    m = M.build("fno", IN, OUT)
    m(torch.randn(1, IN, N, N)).pow(2).mean().backward()
    assert M.clip_grad_norm(m.parameters(), 1.0) > 0.0


def test_the_registry_refuses_an_unknown_name():
    with pytest.raises(ValueError, match="unknown architecture"):
        M.build("transformer", IN, OUT)


def test_width_and_modes_pass_through():
    """A sweep should be a dict, not an edit to models.py."""
    small = M.count_parameters(M.build("fno", IN, OUT, width=8, modes=4))
    big = M.count_parameters(M.build("fno", IN, OUT, width=32, modes=16))
    assert big > 10 * small
