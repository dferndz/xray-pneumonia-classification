import torch
import argparse
from torchvision import transforms
from tqdm.auto import tqdm
import torch.utils.tensorboard as tb
from utils import load_data
from model import Classifier


def train(args):
    log_dir = args.log_dir
    epochs = int(args.epochs)
    lr = float(args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = tb.SummaryWriter(log_dir)
    train_data = load_data("data/train")
    valid_data = load_data("data/test")
    model = Classifier()
    model.to(device)
    global_step = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):
        model.train()
        accuracy_vals = []
        for x, y in train_data:
            x, y = x.to(device).float(), y.to(device).float()

            pred = model(x)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            logger.add_scalar("train/loss", loss, global_step)
            accuracy_vals.append(((pred > 0.5) == y).float().mean().item())
        logger.add_scalar("train/accuracy", torch.FloatTensor(accuracy_vals).mean().item(), global_step)

        model.eval()
        accuracy_vals = []
        for x, y in valid_data:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            accuracy_vals.append(((pred > 0.5) == y).float().mean().item())
        logger.add_scalar("valid/accuracy", torch.FloatTensor(accuracy_vals).mean().item(), global_step)
    
    model.save()


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("-e", "--epochs", default=10)
    parser.add_argument("--lr", default=1e-3)

    args = parser.parse_args()
    train(args)
    