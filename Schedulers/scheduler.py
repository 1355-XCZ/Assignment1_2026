import math

from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


# ── Scheduler factories ──────────────────────────────────────────────────────

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


class _WarmupFactor:
    """Picklable callable for LambdaLR: inverse-exponential (log) warmup then constant.

    Paired with Adam(lr=1.0), the effective lr at step t is:
        warmup phase (t < W):  lr(t) = learning_rate * log(t + 1) / log(W)
        constant phase:        lr(t) = learning_rate

    References:
        - QANet-localminimum / NLPLearn:
              lr = min(0.001, 0.001 / log(999) * log(step + 1))
        - QANet-BangLiu:
              cr = 1 / log(W);  factor = cr * log(step + 1) if step < W else 1
    """
    # [OLD] linear warmup: learning_rate * step / warmup_steps
    # [FIX] inverse-exponential warmup: learning_rate * log(step+1) / log(W)
    def __init__(self, warmup_steps: int, learning_rate: float):
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self._inv_log_w = 1.0 / math.log(warmup_steps) if warmup_steps > 1 else 1.0

    def __call__(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.learning_rate * self._inv_log_w * math.log(step + 1)
        return self.learning_rate


def lambda_scheduler(optimizer, args):
    """LambdaLR with inverse-exponential warmup then constant lr (QANet schedule)."""
    warmup_steps = getattr(args, "warmup_steps", 1000)
    return LambdaLR(optimizer, lr_lambda=_WarmupFactor(warmup_steps, args.learning_rate))


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}
