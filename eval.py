import torch
import argparse
from torchvision import transforms
from tqdm.auto import tqdm
import torch.utils.tensorboard as tb
from utils import load_data
from model import Classifier


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_data = load_data("data/test")
    model = Classifier()
    model.load()
    model.to(device)

    accuracy_vals = []

    for x, y in eval_data:
        x, y = x.to(device), y.to(device)
        pred = model(x) > 0.5

        accuracy_vals.append((pred == y).float().mean().item())

    return torch.FloatTensor(accuracy_vals).mean().item()



if __name__ == "__main__":
    accuracy = evaluate()
    print(f"accuracy: {accuracy}")
    