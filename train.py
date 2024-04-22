from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import transforms
import argparse
from tqdm import tqdm

import random

seed = 42 

torch.manual_seed(seed)


from models.model import CompactCNN
from models.model_rgb import CompactCNN_rgb
from models.model_ds import CompactCNN_ds
from dataset import GazeDataset
from preprocess import preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--preprocess', type=bool, default=True)
parser.add_argument('--no-preprocess', action='store_false', dest='preprocess')
# parser for rgb images
parser.add_argument('--rgb', action='store_true', dest='rgb')
parser.add_argument('--ds', action='store_true', dest='ds')



args = parser.parse_args()

data_path = args.data_path
num_epochs = args.num_epochs

file_name = data_path.split('/')[-1]


if args.rgb:
    net = CompactCNN_rgb()
    size = (192, 1800)
    suffix = '_rgb'
elif args.ds:
    net = CompactCNN_ds()
    size = (48, 450)
    suffix = '_ds'
else:
    net = CompactCNN()
    size = (192, 1800)
    suffix = ''

# preprocess the data
if args.preprocess == True:
    print('Preprocessing the data...' )
    preprocess(data_path, suffix)
    print('Preprocessing complete')

# define the transforms
transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])

# define the dataset
dataset = GazeDataset(
    csv_file=data_path+f'/label_training_{file_name}{suffix}.csv',
    root_dir=data_path+f'/training_{file_name}{suffix}/',
    transform=transform
)


# define the data loader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# define the optimizer
optimizer = optim.Adam(net.parameters(), lr=9e-4)

# define the training loop
def train(net, dataloader, optimizer, epochs=1):
    net.train()
    log_file_path = f'./logs/{file_name}{suffix}.csv'
    with open(log_file_path, 'w') as f:
        f.write('Iteration,Loss\n')
    for epoch in range(epochs):
        loss_100 = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            direction = outputs

            loss = F.mse_loss(direction, labels[..., :2])

            loss.backward()
            optimizer.step()

            loss_100 += loss.item()
            
            if i % 100 == 99:
                # print(f'[{i+1}, {i+1}] loss: {loss_100:.3f}')

                # with open(log_file_path, 'a') as f:
                #     f.write(f'{i+1},{loss_100}\n')

                # loss_100 = 0.0
                print(f'[{i+1}, {i+1}] loss: {loss_100/100:.3f}')
                with open(log_file_path, 'a') as f:
                    f.write(f'{i+1},{loss_100/100}\n')
                loss_100 = 0.0
                


# train the network

print( 'Training the network...')
train(net, dataloader, optimizer, num_epochs)
torch.save(net.state_dict(), f'./networks/gazenet_{file_name}{suffix}.pth')
print(f'Finished Training, network saved as {file_name}{suffix}.pth')