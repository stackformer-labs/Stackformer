import torch

from stackformer.logging.metrics import distributed_mean
from stackformer.metrics import accuracy, f1_score, perplexity, precision, recall


def test_classification_metrics_compute():
    preds = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    targets = torch.tensor([1, 0, 1, 1])
    assert accuracy(preds, targets) == 0.75
    assert round(precision(preds, targets), 4) == 1.0
    assert round(recall(preds, targets), 4) == round(2 / 3, 4)
    assert round(f1_score(preds, targets), 4) == 0.8


def test_perplexity_and_distributed_mean_cpu():
    assert round(perplexity(0.0), 4) == 1.0
    assert distributed_mean(2.5) == 2.5
