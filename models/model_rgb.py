import torch
import torch.nn as nn

class CompactCNN_rgb(nn.Module):
    def __init__(self):
        super(CompactCNN_rgb, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=12, stride=4)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=4)
        self.fc = nn.Linear(in_features=2*1*7, out_features=2)

    def forward(self, x):
        # Checking the input dimensions
        assert x.size()[2:] == (192, 1800), "Input dimensions must be 3x192x1800"
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 1*7*2)
        x = self.fc(x)
        return x
