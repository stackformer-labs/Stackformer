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


def test_module_exports_imports():
    from stackformer.amp import AMPScaler
    from stackformer.distributed import init_distributed
    from stackformer.engine import Trainer
    from stackformer.logging import Logger
    from stackformer.metrics import accuracy
    from stackformer.training import train_loop
    from stackformer.utils import get_device

    assert AMPScaler is not None
    assert init_distributed is not None
    assert Trainer is not None
    assert Logger is not None
    assert train_loop is not None
    assert get_device is not None
    assert accuracy is not None


def test_engine_grad_accumulation_with_amp_cpu_fallback():
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    from stackformer.amp import initialize_scaler

    scaler = initialize_scaler(enabled=True)
    state = TrainingState(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device="cpu",
        config={"criterion": nn.CrossEntropyLoss()},
    )
    engine = Engine(state=state, grad_accum_steps=2)
    engine.train_one_epoch(loader, epoch=0)

    assert scaler.is_enabled is False
    assert state.global_step == 2


def test_engine_logs_only_on_main_process(monkeypatch):
    class DummyMonitor:
        def __init__(self):
            self.logged = 0

        def log(self, _):
            self.logged += 1

    x = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainingState(model=model, optimizer=optimizer, device="cpu", config={"criterion": nn.CrossEntropyLoss()})
    monitor = DummyMonitor()
    engine = Engine(state=state, monitor=monitor, log_every=1)

    monkeypatch.setattr("stackformer.engine.engine.is_main_process", lambda: False)
    engine.train_one_epoch(loader, epoch=0)

    assert monitor.logged == 0
