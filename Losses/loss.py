import torch.nn.functional as F


def qa_nll_loss(p1, p2, y1, y2):
    """QA span loss: log_softmax + nll_loss. Expects raw logits."""
    return F.nll_loss(F.log_softmax(p1, dim=1), y1) + F.nll_loss(F.log_softmax(p2, dim=1), y2)


def qa_ce_loss(p1, p2, y1, y2):
    """QA span loss: cross_entropy (= log_softmax + nll_loss internally). Expects raw logits."""
    return F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)


losses = {
    "qa_nll": qa_nll_loss,
    "qa_ce":  qa_ce_loss,
}
