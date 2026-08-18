import pytest
import torch

from llminfra import ManifoldConstrainedHyperConnection


def test_hyperconnection_shape_constraint_and_gradients():
    module = ManifoldConstrainedHyperConnection(8, hc_mult=4, sinkhorn_iters=20)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    output = module(hidden, torch.randn_like(hidden))
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert module.logits.grad is not None


def test_hyperconnection_mixing_matrix_is_doubly_stochastic():
    module = ManifoldConstrainedHyperConnection(4, hc_mult=3, sinkhorn_iters=30)
    matrix = module.mixing_matrix()
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(3), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs", [{"hidden_size": 0}, {"hidden_size": 4, "hc_mult": 0}]
)
def test_hyperconnection_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        ManifoldConstrainedHyperConnection(**kwargs)
