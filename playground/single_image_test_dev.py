import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd

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

    PATH = '/home/aoqiao/developer_dq/homingdrone-learning/data_gym/test_results/test_031901/gaze_net.pth'
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
    output = torch.atan2(output[0][0], output[0][1])
    output = output.item()
    return output

file_path = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_180124/shifted_180/'
csv_path = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_180124/data.csv'

# add a new column to the csv file called new_prediction
image_info = pd.read_csv(csv_path)

image_info['new_prediction_180'] = 0

for file in os.listdir(file_path):
    if file.endswith('.jpg'):
        img_path = os.path.join(file_path, file)
        pred = predict(img_path)
        idx = image_info[image_info['path_idx'] == int(file.split('_')[0])].index[0]

        image_info.at[idx, 'new_prediction_180'] = pred
        print(pred)

image_info.to_csv(csv_path, index=False)
print('done')