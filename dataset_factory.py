import csv
import os

import torch
from torch.utils.data import DataLoader

from chest_xray_dataloader import ChestXrayDataset


# Builds datasets here based on the configuration
def get_datasets(config):
    image_dir = config["dataset"]["images_root_dir"]
    labels_file = config["dataset"]["labels_path"]
    splits_root_dir = config["dataset"]["data_split"]

    train_split = os.path.join(splits_root_dir, 'train.csv')
    val_split = os.path.join(splits_root_dir, 'val.csv')
    test_split = os.path.join(splits_root_dir, 'test.csv')

    train_data_loader = get_chest_xray_dataloader(
        labels_file, train_split, config, image_dir, train=True)
    val_data_loader = get_chest_xray_dataloader(
        labels_file, val_split, config, image_dir, train=False)
    test_data_loader = get_chest_xray_dataloader(
        labels_file, test_split, config, image_dir, train=False)

    return train_data_loader, val_data_loader, test_data_loader


def get_chest_xray_dataloader(labels_file, data_file, config, image_dir, train):
    dataset = ChestXrayDataset(
        labels_file,
        data_file,
        config,
        image_dir,
        train=train)

    return DataLoader(dataset=dataset,
                      batch_size=config['dataset']['batch_size'],
                      shuffle=True,
                      num_workers=config['dataset']['num_workers'],
                      pin_memory=True)


def _test():
    config = {
        "dataset": {
            "labels_path": "./data/labels.csv",
            "data_split": "./data/balanced",
            "images_root_dir": "/datasets/ChestXray-NIHCC/images/",
            "img_size": 256,
            "batch_size": 64,
            "num_workers": 4
        }
    }
    train_data_loader, val_data_loader, test_data_loader = get_datasets(config)
    train = torch.cat([imgs for _, (imgs, _) in enumerate(train_data_loader)])
    val = torch.cat([imgs for _, (imgs, _) in enumerate(val_data_loader)])
    test = torch.cat([imgs for _, (imgs, _) in enumerate(test_data_loader)])
    
    print(f"Train: Mean={train.mean()}, Std={train.std()}")
    print(f"Val: Mean={val.mean()}, Std={val.std()}")
    print(f"Test: Mean={test.mean()}, Std={test.std()}")


if __name__ == "__main__":
    _test()
