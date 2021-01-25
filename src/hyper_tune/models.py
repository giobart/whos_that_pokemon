import os

from src.model.CustomModelGroupLoss import Siamese_Group, CNN_MODEL_GROUP
import model_configs

def define_model(trial):
    num_classes = 10177
    cnn_model = CNN_MODEL_GROUP.BN_INCEPTION

    if cnn_model == CNN_MODEL_GROUP.MyCNN:
        model_hparams = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True),
            "filter_channels": 4,
            "filter_size": 3,
            "dropout": 0.00,
            "n_hidden1": 4096,
            "n_hidden2": 2048,
            'temperature': 10,
            'num_labeled_points_class': 2,
        }

    elif cnn_model == CNN_MODEL_GROUP.BN_INCEPTION:
        model_hparams = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True),
            'temperature': trial.suggest_int("temp", 10, 60),
            'num_labeled_points_class': 2,
        }
    else:
        raise Exception("cnn_model is not set correctly")


    pre_model = Siamese_Group(hparams=model_hparams,
                          cnn_model=cnn_model,
                          scheduler_params=model_configs.get_group_scheduler_param(cnn_model),
                          nb_classes=num_classes,
                          finetune=True,
                          )

    pre_model = pre_model.load_from_checkpoint(
        checkpoint_path=os.path.join('data', 'checkpoint', 'finetune_inc', 'Group-epoch=09-val_loss=2.29.ckpt'))
    state_dict = pre_model.model.state_dict()

    model = Siamese_Group(hparams=model_configs.get_group_hparam(cnn_model),
                          cnn_model=cnn_model,
                          scheduler_params=model_configs.get_group_scheduler_param(cnn_model),
                          nb_classes=num_classes,
                          finetune=False,
                          cnn_state_dict=state_dict,
                          )
    return model
