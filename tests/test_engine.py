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
