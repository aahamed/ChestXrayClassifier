import torch

from models import ChestXrayClassifier

N_CLASS = 15


# Build and return the model here based on the configuration.
def get_model(config_data):
    model_type = config_data["model"]["model_type"]
    mode = config_data["model"]["mode"]
    img_size = config_data["dataset"]["img_size"]

    model = ChestXrayClassifier(img_size, N_CLASS, mode, model_type)

    return model
