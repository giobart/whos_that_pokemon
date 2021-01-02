from src.model.CustomModelGroupLoss import Siamese_Group, CNN_MODEL_GROUP


def define_model(trial):
    num_classes = 1000
    cnn_model = CNN_MODEL_GROUP.BN_INCEPTION

    if cnn_model == CNN_MODEL_GROUP.MyCNN:
        model_hparams = {
            "lr": 0.001,
            "weight_decay": 1e-5,
            "filter_channels": 4,
            "filter_size": 3,
            "dropout": 0.00,
            "n_hidden1": 4096,
            "n_hidden2": 2048,
            'temperature': 10,
            'num_labeled_points_class': 2,
        }
        scheduler_params = {
            "step_size": 5,
            "gamma": 0.5,
        }

    elif cnn_model == CNN_MODEL_GROUP.BN_INCEPTION:
        model_hparams = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True),
            'temperature': trial.suggest_int("temp", 10, 60),
            'num_labeled_points_class': 2,
        }

        scheduler_params = {
            "step_size": 10,
            "gamma": 0.5,
        }

    model = Siamese_Group(hparams=model_hparams,
                          cnn_model=cnn_model,
                          scheduler_params=scheduler_params,
                          nb_classes=num_classes,
                          finetune=False,
                          weights_path=None,
                          )

    return model
