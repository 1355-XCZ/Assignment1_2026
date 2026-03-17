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


import math


def _constant_factor(_step):
    return 1.0


class _WarmupFactor:
    """Logarithmic warmup following the QANet paper recipe. Picklable."""
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.cr = 1.0 / math.log(warmup_steps)

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.cr * math.log(step + 1)
        return 1.0


def lambda_scheduler(optimizer, args):
    """LambdaLR with logarithmic warmup following the QANet paper recipe."""
    warmup = getattr(args, "lr_warm_up_num", 1000)
    return LambdaLR(optimizer, lr_lambda=_WarmupFactor(warmup))


def none_scheduler(optimizer, args):
    """No-op scheduler — learning rate stays constant."""
    return LambdaLR(optimizer, lr_lambda=_constant_factor)


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
    "none":    none_scheduler,
}
