from src.model.CustomModelGroupLoss import CNN_MODEL_GROUP


def get_group_hparam(cnn_model):
    print(cnn_model)
    if cnn_model == CNN_MODEL_GROUP.MyCNN:
        return {
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
    elif cnn_model == CNN_MODEL_GROUP.BN_INCEPTION:
        return {
            "lr": 0.0001602403,
            "weight_decay": 8.465428e-5,
            'temperature': 12,
            'num_labeled_points_class': 2
        }
    else:
        raise Exception("cnn_model is not defined correctly")


def get_group_scheduler_param(cnn_model):
    if cnn_model == CNN_MODEL_GROUP.MyCNN:
        return {
            "step_size": 5,
            "gamma": 0.5,
        }
    elif cnn_model == CNN_MODEL_GROUP.BN_INCEPTION:
        return {
            "step_size": 10,
            "gamma": 0.5,
        }
