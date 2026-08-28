"""Meta-controller: Dirichlet(α>1) plasticity-budget policy over 4 control axes
(explore, stable, prec, diff), updated by a twin-critic SAC-style objective, and
a budget→RLVR-knob mapping. Adapts the old neuromodulator.py to the v2 design.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

AXES = ("explore", "stable", "prec", "diff")  # (구 NA, 5-HT, ACh, DA)


class DirichletBudgetPolicy(nn.Module):
    """Actor: state -> Dirichlet concentration α>1 over the 4-axis simplex.
    Twin critics: Q(s, w) for the SAC-style update. No γ bootstrapping is used
    (single-cycle contextual bandit over the outer reward); do NOT introduce an
    unused γ (that was a bug in the old code)."""

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, len(AXES)))
        self.q1 = nn.Sequential(nn.Linear(state_dim + len(AXES), hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + len(AXES), hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))

    def alphas(self, s: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.actor(s)) + 1.01           # strictly > 1

    def sample(self, s: torch.Tensor):
        dist = torch.distributions.Dirichlet(self.alphas(s))
        w = dist.rsample()
        return w, dist.log_prob(w)

    def log_prob(self, s: torch.Tensor, w: torch.Tensor):
        dist = torch.distributions.Dirichlet(self.alphas(s))
        return dist.log_prob(w), dist.entropy()

    def q(self, s: torch.Tensor, w: torch.Tensor):
        x = torch.cat([s, w], dim=-1)
        return self.q1(x), self.q2(x)


class MetaController:
    """Wraps the policy with optimizers and an off-policy (per-domain) buffer."""

    def __init__(self, state_dim: int, device="cpu", actor_lr=5e-3, critic_lr=1e-2,
                 entropy_coef=0.05):
        self.net = DirichletBudgetPolicy(state_dim).to(device)
        self.device = device
        self.entropy_coef = entropy_coef
        self.opt_actor = torch.optim.Adam(
            [p for n, p in self.net.named_parameters() if "actor" in n], lr=actor_lr)
        self.opt_critic = torch.optim.Adam(
            [p for n, p in self.net.named_parameters() if n.startswith("q")], lr=critic_lr)
        self.buffers: dict[str, list] = {}

    @torch.no_grad()
    def act(self, state) -> dict:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        w, logp = self.net.sample(s)
        w = w.squeeze(0)
        return {ax: float(w[i]) for i, ax in enumerate(AXES)} | {
            "_w": w.detach().cpu(), "_state": s.squeeze(0).detach().cpu(),
            "_logp": float(logp)}

    def store(self, domain, act, reward):
        self.buffers.setdefault(domain, []).append(
            {"s": act["_state"], "w": act["_w"], "r": float(reward)})

    def update(self, domain, epochs=3):
        """SAC-style: critics regress the (normalized) outer reward; actor
        maximizes min-Q + entropy. No γ (contextual bandit)."""
        buf = self.buffers.get(domain, [])
        if len(buf) < 2:
            return
        rs = torch.tensor([b["r"] for b in buf])
        mu, sd = rs.mean(), rs.std() + 1e-8
        for _ in range(epochs):
            for b in buf:
                s = b["s"].to(self.device).unsqueeze(0)
                w = b["w"].to(self.device).unsqueeze(0)
                r = torch.tensor([[(b["r"] - mu) / sd]], device=self.device)
                # critic
                self.opt_critic.zero_grad()
                q1, q2 = self.net.q(s, w)
                closs = F.mse_loss(q1, r) + F.mse_loss(q2, r)
                closs.backward()
                self.opt_critic.step()
                # actor (freeze critics)
                self.opt_actor.zero_grad()
                w_new, logp = self.net.sample(s)
                for p in [pp for nn_, pp in self.net.named_parameters() if nn_.startswith("q")]:
                    p.requires_grad_(False)
                q1n, q2n = self.net.q(s, w_new)
                aloss = (self.entropy_coef * logp - torch.min(q1n, q2n).squeeze(-1)).mean()
                aloss.backward()
                for p in [pp for nn_, pp in self.net.named_parameters() if nn_.startswith("q")]:
                    p.requires_grad_(True)
                self.opt_actor.step()


def budget_to_knobs(w: dict, base_lr=1e-5) -> dict:
    """Map the 4-axis plasticity budget to RLVR/training knobs (실험설계 §3)."""
    we, ws, wp, wd = w["explore"], w["stable"], w["prec"], w["diff"]
    # v2b fix: floor the forgetting-defense knobs so the meta-controller cannot
    # monopolize precision (high LR) and starve stability (rehearsal/KL), which in
    # v2a caused N-SIAC to forget MORE than fixed budget. Floors keep rehearsal/KL
    # active regardless of the sampled stability budget.
    # Foundational numeric ranges grounded in peer-reviewed literature (see 실험기록.md
    # "실험 수치 Grounding"): sampling temp for RL rollouts ~0.7-1.0 (cap 1.2);
    # RLHF/GRPO KL beta 0.01-0.1, recommended start 0.02; small-model full-FT LR 1e-5-5e-5,
    # AdamW weight decay 0.01; rehearsal fraction per ER/CLEAR/GEM.
    return {
        "temperature": round(min(1.2, 0.6 + 1.0 * we), 3),   # explore -> temp (capped, grounded)
        "entropy_coef": round(0.01 * we, 4),                 # explore -> entropy bonus
        "kl_beta": round(max(0.02, 0.04 * (1.0 - we) + 0.04 * ws), 4),  # KL beta in [0.02,0.1]
        "rehearsal_ratio": round(max(0.1, 0.3 * ws), 3),     # stable -> rehearsal (floor 0.1)
        "weight_decay": round(0.01 * (1.0 + 5.0 * ws), 4),   # base 0.01 (AdamW norm)
        "learning_rate": round(base_lr * (1.0 + 1.5 * wp), 8),  # LR in [1e-5, 2.5e-5]
        "difficulty": round(wd, 3),                          # diff -> ZPD target
        # stable emphasized (> uniform 0.25) -> enable TRL soft reference sync
        # (ref_model_mixup_alpha EMA toward current policy; a "soft reset", not hard).
        # Threshold 0.35 (not 0.6): on a 4-simplex a single share rarely exceeds 0.6,
        # which left this sub-axis near-dead; 0.35 means "stability above uniform".
        "ref_reset": bool(ws > 0.35),
    }
