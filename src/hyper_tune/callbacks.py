from pytorch_lightning import Callback
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def get_callbacks(trial):
    metrics_callback = MetricsCallback()
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_R%_@1',
        patience=5,
        strict=True,
        verbose=False,
        mode='max'
    )
    return [metrics_callback, early_stop_callback, PyTorchLightningPruningCallback(trial, monitor="val_R%_@1")]
