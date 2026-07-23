from pathlib import Path
import warnings
from tests._test_utils import _checkpoint

from stackformer.logging import CSVLogger, Logger


def test_csv_logger_writes_metrics(tmp_path):
    _checkpoint("test_csv_logger_writes_metrics setup", tmp_path=str(tmp_path))
    logger = CSVLogger(log_dir=str(tmp_path), filename="metrics.csv")
    logger.log({"loss": 1.0, "lr": 1e-3})
    logger.flush()
    logger.close()

    _checkpoint("Reading CSV logger output file")
    content = (tmp_path / "metrics.csv").read_text()
    assert "loss" in content
    assert "0.001" in content


def test_unified_logger_uses_csv_backend(tmp_path):
    _checkpoint("test_unified_logger_uses_csv_backend setup", tmp_path=str(tmp_path))
    logger = Logger(csv=True, tensorboard=False, wandb=False, log_dir=str(tmp_path), experiment_name="run")
    logger.log({"loss": 0.5})
    logger.close()
    _checkpoint("Asserting run_metrics.csv exists")
    assert list(Path(tmp_path).glob("run_metrics.csv"))


def test_unified_logger_warns_once_on_backend_failure():
    _checkpoint("test_unified_logger_warns_once_on_backend_failure setup")
    class FailingBackend:
        def __init__(self):
            self.calls = 0

        def log(self, metrics):
            self.calls += 1
            raise RuntimeError("boom")

    logger = Logger(csv=False, tensorboard=False, wandb=False)
    backend = FailingBackend()
    logger.backends.append(backend)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        logger.log({"loss": 1.0})
        logger.log({"loss": 0.5})

    runtime_warnings = [w for w in records if issubclass(w.category, RuntimeWarning)]
    _checkpoint("Asserting runtime warning issued once", warning_count=len(runtime_warnings))
    assert len(runtime_warnings) == 1
    assert backend.calls == 1
