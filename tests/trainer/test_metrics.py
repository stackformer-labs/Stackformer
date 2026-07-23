import torch
from tests._test_utils import _checkpoint

from stackformer.logging.metrics import distributed_mean
from stackformer.metrics import accuracy, f1_score, perplexity, precision, recall


def test_classification_metrics_compute():
    _checkpoint("test_classification_metrics_compute setup")
    preds = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    targets = torch.tensor([1, 0, 1, 1])

    acc = accuracy(preds, targets)
    prec = precision(preds, targets)
    rec = recall(preds, targets)
    f1 = f1_score(preds, targets)

    _checkpoint("Asserting classification metrics values", accuracy=acc, precision=prec, recall=rec, f1=f1)
    assert acc == 0.75
    assert round(prec, 4) == 1.0
    assert round(rec, 4) == round(2 / 3, 4)
    assert round(f1, 4) == 0.8


def test_perplexity_and_distributed_mean_cpu():
    _checkpoint("test_perplexity_and_distributed_mean_cpu setup")
    ppl = perplexity(0.0)
    d_mean = distributed_mean(2.5)
    _checkpoint("Asserting perplexity and distributed_mean", ppl=ppl, d_mean=d_mean)
    assert round(ppl, 4) == 1.0
    assert d_mean == 2.5
