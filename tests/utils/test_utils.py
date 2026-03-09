from stackformer.amp import AMPScaler
from stackformer.distributed import init_distributed
from stackformer.engine import Trainer
from stackformer.logging import Logger
from stackformer.training import train_loop
from stackformer.utils import get_device


def test_module_exports_imports():
    assert AMPScaler is not None
    assert init_distributed is not None
    assert Trainer is not None
    assert Logger is not None
    assert train_loop is not None
    assert get_device is not None
