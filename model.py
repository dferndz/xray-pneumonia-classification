import torch
from torchvision import transforms
import torchvision


MODEL_STATE_PATH = "classifier.pth"


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, bias=False)
        self.net = torchvision.models.resnet34(pretrained=True)
        self.net.fc = torch.nn.Linear(512, 1, bias=True)
        self.eval()
    
    def forward(self, x):
        return self.net(self.conv(x)).squeeze(1)
    
    def save(self):
        torch.save(self.state_dict(), MODEL_STATE_PATH)
    
    def load(self):
        self.load_state_dict(torch.load(MODEL_STATE_PATH))


if __name__ == "__main__":
    model = Classifier()
    model.save()
    print(model)