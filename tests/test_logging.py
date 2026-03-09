from pathlib import Path

from stackformer.logging import CSVLogger, Logger


def test_csv_logger_writes_metrics(tmp_path):
    logger = CSVLogger(log_dir=str(tmp_path), filename="metrics.csv")
    logger.log({"loss": 1.0, "lr": 1e-3})
    logger.flush()
    logger.close()

    content = (tmp_path / "metrics.csv").read_text()
    assert "loss" in content
    assert "0.001" in content


def test_unified_logger_uses_csv_backend(tmp_path):
    logger = Logger(csv=True, tensorboard=False, wandb=False, log_dir=str(tmp_path), experiment_name="run")
    logger.log({"loss": 0.5})
    logger.close()

    files = list(Path(tmp_path).glob("run_metrics.csv"))
    assert files
