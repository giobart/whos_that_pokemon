import pytorch_lightning as pl
import torch
from . import callbacks
from .models import define_model
import multiprocessing as mp
from src.modules.classification_data_model import Classification_Model, DATASETS

def objective(trial):
    model = define_model(trial)

    num_classes_iter = 24
    num_elem_class = 2
    batch_size = num_classes_iter * num_elem_class
    dataloader = Classification_Model(name=DATASETS.CELEBA,
                                   nb_classes=model.nb_classes,
                                   class_split=True,
                                   batch_size=batch_size,
                                   num_classes_iter=num_classes_iter,
                                   splitting_points=(0.20, 0),
                                   input_shape=(3, model.input_size, model.input_size),
                                   num_workers=mp.cpu_count(),
                                   finetune=False)

    dataloader.setup()

    calls = callbacks.get_callbacks(trial)
    trainer_params = {
        "check_val_every_n_epoch": 1,
        "callbacks": calls,
        "fast_dev_run": False,
        "max_epochs": 3,
        "gpus": 1 if torch.cuda.is_available() else None,

    }

    trainer = pl.Trainer(**trainer_params, logger=False)
    trainer.fit(model, dataloader)

    return calls[0].metrics[-1]["val_R%_@1"].item()