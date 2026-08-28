"""Verify the paper's bf16 weight-decay inertness claim (sec_method):
a parameter at the RMS magnitude (0.042) moves exactly zero under 720 AdamW
decay applications, and the trajectory is bit-identical for lambda=0 vs 0.06.
Pure-decay isolation: grad=0 so the update reduces to theta *= (1 - lr*lambda),
executed in bfloat16 exactly as AdamW's decoupled decay does."""
import torch

def run(lam, steps=720, lr=1e-5, val=0.042):
    init = torch.tensor([val], dtype=torch.bfloat16)  # nearest-representable bf16 start
    th = init.clone()
    for _ in range(steps):
        th = (th.float() * (1.0 - lr * lam)).to(torch.bfloat16)  # decoupled decay, round-to-nearest
    return init, th

(i0, t0), (i6, t6) = run(0.0), run(0.06)
same_bits = torch.equal(t0.view(torch.int16), t6.view(torch.int16))
moved = (t6.float() - i6.float()).abs().item()  # displacement vs the bf16 INITIAL value
print(f"lambda=0    final: {t0.item():.10f}")
print(f"lambda=0.06 final: {t6.item():.10f}")
print(f"bit-identical: {same_bits}  |  displacement at lambda=0.06: {moved:.3e}")
fp32 = 0.042
for _ in range(720): fp32 *= (1.0 - 1e-5 * 0.06)
print(f"fp32 reference displacement: {abs(fp32-0.042):.3e} (nonzero, as expected)")
assert same_bits and moved == 0.0, "PAPER CLAIM VIOLATED"
print("PASS: bf16 decay inert; bit-identical across lambda — matches sec_method claim")
