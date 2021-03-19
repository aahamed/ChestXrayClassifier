import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader  # For custom data-sets

LABEL_ENCODINGS = {
    "Atelectasis": 0,
    "Consolidation": 1,
    "Infiltration": 2,
    "Pneumothorax": 3,
    "Edema": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Effusion": 7,
    "Pneumonia": 8,
    "Pleural_Thickening": 9,
    "Cardiomegaly": 10,
    "Nodule": 11,
    "Mass": 12,
    "Hernia": 13,
    "No Finding": 14,
}

N_CLASSES = len(LABEL_ENCODINGS.keys())


def one_hot_encode_labels(labels):
    one_hot = torch.zeros(N_CLASSES)
    for label in labels:
        idx = LABEL_ENCODINGS[label]
        one_hot[idx] = 1
    return one_hot


class ChestXrayDataset(Dataset):
    def __init__(self, labels_file, data_file, config, image_dir,
                 transforms_=None, train=False):
        self.labels = pd.read_csv(labels_file, index_col=0, usecols=[0, 1])
        self.data = pd.read_csv(data_file, header=None)
        self.config = config
        self.image_dir = image_dir
        self.mode = labels_file
        self.train = train

        img_size = config["dataset"]["img_size"]
        self.resize = transforms.Compose([
            transforms.Resize(img_size, interpolation=2),
            transforms.CenterCrop(img_size)
        ])

        # The following transformation normalizes each channel using the mean and std provided
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49,), (0.25,)), ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 0]
        labels = self.labels.loc[file_name][0].split('|')
        file_name = self.image_dir + file_name

        # some images don"t seem to be grayscale ?
        # for ex. /datasets/ChestXray-NIHCC/images/00000004_000.png
        img = Image.open(file_name).convert("L")

        # apply transformations
        img = self.apply_transform(img)
        img = np.asarray(img) / 255.  # scaling [0-255] values to [0-1]
        img = self.normalize(img).float()
        labels = one_hot_encode_labels(labels)

        return img, labels

    def apply_transform(self, img):
        img = self.resize(img)

        # no augmentation for val, test data
        if not self.train:
            return img

        # apply data augmentation for train data
        r = np.random.randint(1, 100)
        if 1 <= r <= 15:
            img = transforms.functional.hflip(img)
        elif 16 <= r <= 30:
            img = transforms.functional.vflip(img)
        elif 31 <= r <= 45:
            angle = np.random.randint(-15, 15)
            img = transforms.functional.rotate(img, angle, fill=(0,))
            # The PIL version on my dsmlp account doesn't like fill
            # img = transforms.functional.rotate(img, angle)

        return img


def _test():
    labels_file = "./data/labels.csv"
    data_file = "./data/balanced/train.csv"
    image_dir = "/datasets/ChestXray-NIHCC/images/"
    img_size = 256
    batch_size = 4
    num_ch = 1
    config = {"dataset": {"img_size": img_size}}
    train = False
    xray_dataset = ChestXrayDataset(labels_file, data_file,
                                    config, image_dir, train=train)
    train_loader = DataLoader(dataset=xray_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=False)
    train_loader_iter = iter(train_loader)
    for i in range(10):
        imgs, labels = next(train_loader_iter)
        print(f'i: {i} imgs shape: {imgs.shape} labels shape: {labels.shape}')
        assert imgs.shape == ( batch_size, num_ch, img_size, img_size )
        assert labels.shape == ( batch_size, N_CLASSES )
    print('Test Passed!')


if __name__ == "__main__":
    _test()
