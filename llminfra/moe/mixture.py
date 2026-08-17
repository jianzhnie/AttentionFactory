"""Educational mixture-of-experts (MoE) modules.

These modules implement the routing and expert computation patterns used by
mainstream MoE models such as Qwen3, DeepSeekMoE, Mixtral, DBRX, Baichuan-M3
and Nemotron-3. They are teaching implementations: expert execution loops
over expert ids instead of using optimized group GEMM kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExpertFFN(nn.Module):
    """One expert's feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if activation not in {"silu", "relu", "gelu"}:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_name = activation
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self._activation(self.w1(x)))

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "silu":
            return F.silu(x)
        if self.activation_name == "relu":
            return F.relu(x)
        return F.gelu(x)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)


class TopKRouter(nn.Module):
    """Top-k expert router with optional training-time noise.

    Args:
        hidden_size: Input feature dimension.
        num_experts: Number of routed experts.
        top_k: Number of experts selected per token.
        add_noise: Enable the Switch/GShard-style noise used during training.
        noise_epsilon: Scale of the routing noise.
        dropout: Dropout applied to routing weights.
        scoring_func: Scoring applied to the selected experts' logits,
            ``"softmax"`` (default) or ``"sigmoid"``. With ``"sigmoid"`` the
            weights are ``sigmoid(logits)`` renormalized to sum to 1.
        aux_free_balance: Enable a DeepSeek-V3-style auxiliary-loss-free
            balancing bias. A gradient-free ``router_bias`` parameter is
            registered; top-k selection uses ``logits + router_bias`` while
            the routing weights are still computed from the unbiased logits.
            After each training-mode forward the bias is updated by a
            discrete step proportional to the sign of each expert's load
            violation. This is a teaching simplification: the real systems
            update the bias per training step with a tuned schedule.
        balance_update_rate: Step size ``u`` of the bias update.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        add_noise: bool = False,
        noise_epsilon: float = 1e-2,
        dropout: float = 0.0,
        scoring_func: str = "softmax",
        aux_free_balance: bool = False,
        balance_update_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        if scoring_func not in {"softmax", "sigmoid"}:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.add_noise = bool(add_noise)
        self.noise_epsilon = float(noise_epsilon)
        self.scoring_func = scoring_func
        self.aux_free_balance = bool(aux_free_balance)
        self.balance_update_rate = float(balance_update_rate)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.noise_proj = (
            nn.Linear(hidden_size, num_experts, bias=False) if add_noise else None
        )
        self.dropout = nn.Dropout(dropout)
        if aux_free_balance:
            # Gradient-free on purpose: balanced by discrete updates, not SGD.
            self.router_bias = nn.Parameter(
                torch.zeros(num_experts), requires_grad=False
            )
        else:
            self.register_parameter("router_bias", None)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(routing_weights, expert_indices)``.

        ``routing_weights`` has shape ``(batch, top_k)`` and
        ``expert_indices`` has shape ``(batch, top_k)``. With ``dropout=0``
        the weights sum to 1 along the last dimension (dropout during
        training rescales them, as usual).
        """
        logits = self.router(x)
        if self.training and self.add_noise and self.noise_proj is not None:
            # Switch/GShard-style: noise scaled by a learned per-expert std
            # and the configured epsilon. A constant shift would not change
            # the top-k ranking, so the epsilon must multiply the noise.
            noise = torch.randn_like(logits)
            noise = noise * F.softplus(self.noise_proj(x))
            logits = logits + noise * self.noise_epsilon

        # The balancing bias steers *which* experts are picked, but never
        # enters the weights (DeepSeek-V3 auxiliary-loss-free balancing).
        selection_logits = logits
        if self.router_bias is not None:
            selection_logits = logits + self.router_bias
        _, indices = torch.topk(selection_logits, self.top_k, dim=-1)
        weights = logits.gather(-1, indices)
        weights = self._score(weights)
        weights = self.dropout(weights)

        if self.training and self.router_bias is not None:
            self._update_router_bias(indices)
        return weights, indices

    def _score(self, selected_logits: torch.Tensor) -> torch.Tensor:
        """Turn selected experts' logits into normalized routing weights."""
        if self.scoring_func == "softmax":
            return torch.softmax(selected_logits, dim=-1)
        weights = torch.sigmoid(selected_logits)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def _update_router_bias(self, indices: torch.Tensor) -> None:
        """Step the bias by ``u * sign(target_load - load)`` per expert."""
        counts = torch.bincount(
            indices.reshape(-1), minlength=self.num_experts
        ).to(torch.float32)
        target = indices.numel() / self.num_experts
        violation = target - counts
        self.router_bias.add_(self.balance_update_rate * torch.sign(violation))

    def routing_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw router logits without training noise."""
        return self.router(x)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, add_noise={self.add_noise}, "
            f"scoring_func={self.scoring_func}, "
            f"aux_free_balance={self.aux_free_balance}"
        )


class ExpertChoiceRouter(nn.Module):
    """Expert-choice routing: each expert selects its top tokens.

    Instead of each token picking its top-k experts, each expert picks its
    top-``top_tokens`` tokens, which guarantees perfectly balanced expert
    loads by construction (every expert receives the same number of tokens).

    Shape convention: ``forward`` returns ``(weights, token_indices)`` where
    both have shape ``(num_experts, top_tokens)``. ``token_indices[e, j]`` is
    the index (into the flattened token axis of the input) of the j-th token
    chosen by expert ``e``, and ``weights[e, j]`` is that token's routing
    weight for expert ``e``. Weights come from a softmax over the token axis,
    so for each expert the weights over all tokens (not only the chosen ones)
    sum to 1. A token may be chosen by several experts or by none.

    Args:
        hidden_size: Input feature dimension.
        num_experts: Number of routed experts.
        top_tokens: Number of tokens each expert selects.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_tokens: int,
    ) -> None:
        super().__init__()
        if top_tokens < 1:
            raise ValueError("top_tokens must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_tokens = int(top_tokens)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weights, token_indices)``, each ``(num_experts, top_tokens)``.

        Args:
            x: Token features of shape ``(num_tokens, hidden_size)`` with
                ``num_tokens >= top_tokens``.
        """
        logits = self.router(x)  # (num_tokens, num_experts)
        expert_scores = logits.transpose(0, 1)  # (num_experts, num_tokens)
        scores = torch.softmax(expert_scores, dim=-1)
        weights, token_indices = torch.topk(scores, self.top_tokens, dim=-1)
        return weights, token_indices

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"top_tokens={self.top_tokens}"
        )


class MixtureOfExperts(nn.Module):
    """Standard top-k routed mixture of experts.

    The module does not add a residual connection; callers should apply the
    transformer residual outside this layer.

    Args:
        expert_dropout: Probability of dropping a token's routed expert
            weight during training. Dropped weights are zeroed and the
            surviving experts of that token are renormalized to sum to 1
            (tokens whose experts were all dropped contribute nothing this
            step). Teaching simplification of expert-level dropout.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        top_k: int = 2,
        activation: str = "silu",
        add_router_noise: bool = False,
        bias: bool = True,
        expert_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= expert_dropout < 1.0:
            raise ValueError("expert_dropout must satisfy 0 <= p < 1")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.expert_dropout = float(expert_dropout)
        self.experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size, activation, bias)
            for _ in range(num_experts)
        )
        self.router = TopKRouter(
            hidden_size,
            num_experts,
            top_k=top_k,
            add_noise=add_router_noise,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to experts and return the combined output."""
        flat = x.reshape(-1, self.hidden_size)
        routing_weights, expert_indices = self.router(flat)
        if self.training and self.expert_dropout > 0.0:
            keep = torch.rand_like(routing_weights) >= self.expert_dropout
            routing_weights = routing_weights * keep
            totals = routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights / totals.clamp_min(1e-12)
        output = torch.zeros_like(flat)

        for k in range(self.top_k):
            token_weights = routing_weights[:, k]
            token_experts = expert_indices[:, k]
            for expert_id in range(self.num_experts):
                mask = token_experts == expert_id
                if not mask.any():
                    continue
                selected = flat[mask]
                output[mask] += token_weights[mask].unsqueeze(-1) * self.experts[
                    expert_id
                ](selected)
        return output.view_as(x)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"intermediate_size={self.intermediate_size}, top_k={self.top_k}, "
            f"expert_dropout={self.expert_dropout}"
        )


class DeepSeekMoE(nn.Module):
    """DeepSeek-style MoE with a small set of shared experts.

    The output is ``routed_experts(x) + sum(shared_experts(x))``. A residual
    connection is intentionally left to the transformer block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        intermediate_size: int,
        top_k: int = 6,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_shared_experts < 1:
            raise ValueError("num_shared_experts must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_routed_experts = int(num_routed_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.routed = MixtureOfExperts(
            hidden_size,
            num_routed_experts,
            intermediate_size,
            top_k=top_k,
            activation=activation,
            bias=bias,
        )
        self.shared_experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size, activation, bias)
            for _ in range(num_shared_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return routed output plus shared-expert output."""
        shared = torch.stack([expert(x) for expert in self.shared_experts]).sum(dim=0)
        return self.routed(x) + shared

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_routed_experts={self.num_routed_experts}, "
            f"num_shared_experts={self.num_shared_experts}, "
            f"intermediate_size={self.intermediate_size}, top_k={self.top_k}"
        )


class ExpertParallelMoE(nn.Module):
    """Simulated expert-parallel MoE with local expert ownership.

    Each rank owns ``num_experts // world_size`` experts. The router still
    sees the full expert set, but this rank only computes its local experts.
    No real all-reduce or communication is implemented.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        top_k: int = 2,
        world_size: int = 1,
        rank: int = 0,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if world_size < 1 or rank >= world_size:
            raise ValueError("world_size must be >= 1 and rank < world_size")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.local_expert_ids = list(
            range(self.rank, self.num_experts, self.world_size)
        )
        self.experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size, activation, bias)
            for _ in self.local_expert_ids
        )
        self.router = TopKRouter(
            hidden_size,
            num_experts,
            top_k=top_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute outputs for experts owned by this rank."""
        flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.router(flat)
        output = torch.zeros_like(flat)
        for local_index, expert_id in enumerate(self.local_expert_ids):
            token_weights = torch.zeros(
                flat.size(0), device=flat.device, dtype=flat.dtype
            )
            for k in range(self.top_k):
                token_weights += torch.where(
                    indices[:, k] == expert_id,
                    weights[:, k],
                    torch.zeros_like(weights[:, k]),
                )
            mask = token_weights > 0
            if mask.any():
                output[mask] += token_weights[mask].unsqueeze(-1) * self.experts[
                    local_index
                ](flat[mask])
        return output.view_as(x)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"world_size={self.world_size}, rank={self.rank}, "
            f"local_expert_ids={self.local_expert_ids}"
        )


def load_balance_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Auxiliary load-balancing loss for top-k routing.

    Args:
        router_logits: Shape ``(num_tokens, num_experts)``.
        expert_indices: Shape ``(num_tokens, top_k)``.
        num_experts: Number of routed experts.
    """
    if router_logits.size(-1) != num_experts:
        raise ValueError("router_logits last dim must equal num_experts")
    if expert_indices.dim() != 2:
        raise ValueError("expert_indices must have shape (num_tokens, top_k)")
    counts = torch.zeros(num_experts, device=expert_indices.device, dtype=torch.float32)
    for k in range(expert_indices.size(-1)):
        counts += torch.bincount(
            expert_indices[:, k].flatten(),
            minlength=num_experts,
        ).to(counts.dtype)
    fraction = counts / max(1, expert_indices.numel())
    probabilities = torch.softmax(router_logits, dim=-1).mean(dim=0)
    return num_experts * (fraction * probabilities).sum()


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss: penalizes large-magnitude routing logits.

    Introduced by ST-MoE and used by many production MoE models to keep
    router logits in a numerically stable range. Defined as
    ``mean(logsumexp(logits, dim=-1) ** 2)``.

    Args:
        router_logits: Raw router logits of shape ``(num_tokens, num_experts)``.

    Returns:
        Scalar z-loss tensor.
    """
    log_z = torch.logsumexp(router_logits, dim=-1)
    return (log_z * log_z).mean()
