"""
ema.py — Exponential Moving Average for model parameters.

Reference: QANet-BangLiu/model/modules/ema.py
Paper: "We also apply exponential moving average on all trainable
       variables with a decay rate of 0.9999."
"""


class EMA:
    """Maintains shadow copies of trainable parameters as exponential
    moving averages.  Call ``assign`` / ``resume`` to swap EMA weights
    into the model for evaluation and back for training."""

    def __init__(self, decay: float):
        self.decay = decay
        self.shadow = {}
        self.original = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model, num_updates: int):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    decay * self.shadow[name] + (1.0 - decay) * param.data
                )

    def assign(self, model):
        """Swap EMA weights into the model (for evaluation / saving)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original training weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original:
                param.data = self.original[name]
        self.original = {}
