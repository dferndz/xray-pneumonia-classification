import torch
from torchvision import transforms
import torchvision


MODEL_STATE_PATH = "classifier.pth"


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet34(pretrained=True)
        self.net.fc = torch.nn.Linear(512, 1, bias=True)
        self.eval()

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])
    
    def forward(self, x):
        x = self.transform(x)
        return self.net(x).squeeze(1)
    
    def save(self):
        torch.save(self.state_dict(), MODEL_STATE_PATH)
    
    def load(self):
        self.load_state_dict(torch.load(MODEL_STATE_PATH))


if __name__ == "__main__":
    model = Classifier()
    print(model)