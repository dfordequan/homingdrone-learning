import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms

class CompactCNN(nn.Module):
    def __init__(self):
        super(CompactCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=12, stride=4)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=4)
        self.fc = nn.Linear(in_features=2*1*7, out_features=2)

    def forward(self, x):
        assert x.size()[2:] == (192, 1800), "Input dimensions must be 1x201x1800"
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 1*7*2)
        x = self.fc(x)
        return x
    

def predict(img_path):
    net = CompactCNN()

    PATH = '/home/aoqiao/developer_dq/homingdrone-learning/networks/gazenet_.pth'
    net.load_state_dict(torch.load(PATH))

    # img_path = '/home/aoqiao/developer_dq/homingdrone-learning/data_gym/test_combined/home_2_rected.jpg'
    image = Image.open(img_path).convert('L')


    transform = transforms.Compose([
        transforms.Resize((192, 1800)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    output = net(image)
    # return output, output[0][1], output[0][0] 

    output = torch.atan2(output[0][1], output[0][0])
    output = output.item()
    return output

print("13:",predict('/home/aoqiao/developer_dq/homingdrone-learning/data/20240403_094112_rectilinear/13_rected_20240403_094112.jpg'))
print("19:",predict('/home/aoqiao/developer_dq/homingdrone-learning/data/20240403_094112_rectilinear/19_rected_20240403_094112.jpg'))
print("19_shifted_10:",predict('/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240403/20240403_094112_rectilinear/20240403_094112_shifted_10/19_rected_20240403_094112.jpg'))
print("19_photoshop2:",predict('/home/aoqiao/Downloads/19_rected_20240403_094112_ps3.jpg'))
print("19_shifted_90:",predict('/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240403/20240403_095708_rectilinear/shifted_90/19_rected_20240403_095708.jpg'))