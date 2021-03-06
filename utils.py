import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import os
from os import listdir
from PIL import Image
import torchvision


LABELS = ["normal", "pneumonia"]
VB_LABELS = ["virus", "bacteria"]


class XRayDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        #transforms.ColorJitter(0.9, 0.8, 0.2)
    ])

    exclude = [
        ".DS_Store"
    ]

    def __init__(self, path):
        super().__init__()

        normal = os.path.join(path, "NORMAL")
        pneumonia = os.path.join(path, "PNEUMONIA")

        normal_data = [(os.path.join(normal, f), 0) for f in listdir(normal) if f not in self.exclude]
        pneumonia_data = [(os.path.join(pneumonia, f), 1) for f in listdir(pneumonia) if f not in self.exclude]

        self.data = normal_data + pneumonia_data

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = Image.open(image)
        image.load()
        image = self.transform(image)
        return (image, label)


class XRayVBDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        #transforms.ColorJitter(0.9, 0.8, 0.2)
    ])

    exclude = [
        ".DS_Store"
    ]

    @staticmethod
    def get_label(path):
        return 1 if "bacteria" in path else 0

    def __init__(self, path):
        super().__init__()

        pneumonia = os.path.join(path, "PNEUMONIA")

        self.data = [(os.path.join(pneumonia, f), self.get_label(f)) for f in listdir(pneumonia) if f not in self.exclude]

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = Image.open(image)
        image.load()
        image = self.transform(image)
        return (image, label)


def load_data(path, dataset_class=XRayDataset, batch_size=64):
    dataset = dataset_class(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


dataset_factory = {
    "xray": XRayDataset,
    "vb": XRayVBDataset
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="xray")
    args = parser.parse_args()

    data_train = load_data("data/train", dataset_class=dataset_factory[args.dataset])

    for x, y in data_train:
        grid = torchvision.utils.make_grid(x[:8])
        plt.imshow(grid.permute(1,2,0))
        plt.show()
        break
