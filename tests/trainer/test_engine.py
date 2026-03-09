import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import Engine, TrainingState


def test_engine_train_one_epoch_updates_state():
    x = torch.randn(12, 4)
    y = torch.randint(0, 3, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainingState(model=model, optimizer=optimizer, device="cpu", config={"criterion": nn.CrossEntropyLoss()})
    engine = Engine(state=state, max_train_steps=2)

    engine.train_one_epoch(loader, epoch=0)
    assert state.global_step == 2


def test_engine_grad_accumulation_amp_cpu_fallback():
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    from stackformer.amp import initialize_scaler

    scaler = initialize_scaler(enabled=True)
    state = TrainingState(model=model, optimizer=optimizer, scaler=scaler, device="cpu", config={"criterion": nn.CrossEntropyLoss()})
    engine = Engine(state=state, grad_accum_steps=2)
    engine.train_one_epoch(loader, epoch=0)

    assert scaler.is_enabled is False
    assert state.global_step == 2
