"""Train-split rehearsal harvest (v2j) — removes the eval-set re-exposure confound.

Why this exists. The published (v2h/v2i) rehearsal buffer is filled from each domain's
own *evaluation* rollouts, so past-domain retention is measured on prompts the rehearsal
pass trained on (disclosed in the paper as an upper bound mixing transfer with
re-exposure). The clean source must (a) come from the TRAIN split only, (b) be the same
mechanism in every condition, and (c) add no extra generation cost. The inner training
loop's own generations satisfy all three: every GRPO condition scores its rollouts
through the reward function, and the STaR/SFT baseline scores its self-distillation
samples through the same verifiers — so wrapping the reward call captures
verifier-correct (prompt + completion) traces from train prompts for free.

Deduplication matches the published path exactly (120-char completion prefix,
<= 2 distinct solutions per prompt) so the only protocol difference is the split the
traces come from.
"""
from __future__ import annotations


class TrainHarvest:
    """Wraps a TRL reward function; captures verifier-correct traces as a side effect."""

    def __init__(self, per_prompt: int = 2, cap: int = 60):
        self.per_prompt = per_prompt
        self.cap = cap
        self.reset()

    def reset(self):
        self._per_prompt_keys: dict[str, set] = {}
        self._per_prompt_n: dict[str, int] = {}
        self.traces: list[str] = []

    def wrap(self, reward_fn):
        def wrapped(prompts, completions, **cols):
            rewards = reward_fn(prompts, completions, **cols)
            if len(self.traces) < self.cap:
                for p, c, r in zip(prompts, completions, rewards):
                    if r is None or not r > 0:
                        continue
                    if self._per_prompt_n.get(p, 0) >= self.per_prompt:
                        continue
                    k = c[:120]
                    keys = self._per_prompt_keys.setdefault(p, set())
                    if k in keys:
                        continue
                    keys.add(k)
                    self._per_prompt_n[p] = self._per_prompt_n.get(p, 0) + 1
                    self.traces.append(p + c)
                    if len(self.traces) >= self.cap:
                        break
            return rewards

        # preserve TRL's reward-function naming (used in its logged metric keys)
        wrapped.__name__ = getattr(reward_fn, "__name__", "reward_fn")
        return wrapped
