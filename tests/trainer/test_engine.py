import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tests._test_utils import _checkpoint

from stackformer.engine import Engine, TrainingState


def test_engine_train_one_epoch_updates_state(torch_device):
    _checkpoint("test_engine_train_one_epoch_updates_state setup", device=torch_device)
    x = torch.randn(12, 4, device=torch_device)
    y = torch.randint(0, 3, (12,), device=torch_device)
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    model = nn.Linear(4, 3, device=torch_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainingState(model=model, optimizer=optimizer, device=str(torch_device), config={"criterion": nn.CrossEntropyLoss()})
    engine = Engine(state=state, max_train_steps=2)

    _checkpoint("Executing train_one_epoch")
    engine.train_one_epoch(loader, epoch=0)
    _checkpoint("Asserting global_step updated", global_step=state.global_step)
    assert state.global_step == 2


def test_engine_grad_accumulation_amp_cpu_fallback(torch_device):
    _checkpoint("test_engine_grad_accumulation_amp_cpu_fallback setup", device=torch_device)
    x = torch.randn(8, 4, device=torch_device)
    y = torch.randint(0, 3, (8,), device=torch_device)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    model = nn.Linear(4, 3, device=torch_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    from stackformer.amp import initialize_scaler

    scaler = initialize_scaler(enabled=True)
    state = TrainingState(model=model, optimizer=optimizer, scaler=scaler, device=str(torch_device), config={"criterion": nn.CrossEntropyLoss()})
    engine = Engine(state=state, grad_accum_steps=2)

    _checkpoint("Executing train_one_epoch with scaler fallback check")
    engine.train_one_epoch(loader, epoch=0)

    _checkpoint("Asserting CPU scaler disabled and steps counted", is_enabled=scaler.is_enabled, global_step=state.global_step)
    assert scaler.is_enabled is False
    assert state.global_step == 2


class _TupleOutputModel(nn.Module):
    """Auxiliary model returning (logits, loss) tuple to test loss unpacking heuristic (H-04)."""
    def __init__(self, embed_dim=4, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, y=None):
        logits = self.fc(x)
        if y is not None:
            loss = nn.functional.cross_entropy(logits, y)
            return logits, loss
        return logits


def test_engine_loss_heuristic_tuple_output(torch_device):
    """Cover H-04 issue verifying engine handling of tuple outputs."""
    _checkpoint("test_engine_loss_heuristic_tuple_output setup", device=torch_device)
    x = torch.randn(6, 4, device=torch_device)
    y = torch.randint(0, 3, (6,), device=torch_device)
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    model = _TupleOutputModel().to(torch_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainingState(model=model, optimizer=optimizer, device=str(torch_device), config={})
    engine = Engine(state=state, max_train_steps=1)

    _checkpoint("Executing train_one_epoch with tuple output model")
    engine.train_one_epoch(loader, epoch=0)
    assert state.global_step == 1
