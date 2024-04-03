import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd
from models.model import CompactCNN
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--visualize', type=bool, default=False)
args = parser.parse_args()

model_path = args.model_path

model = model_path.split('/')[-1][:-4]


net = CompactCNN()
net.load_state_dict(torch.load(model_path))
    

def predict(img_path):
    
    image = Image.open(img_path).convert('L')


    transform = transforms.Compose([
        transforms.Resize((192, 1800)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    output = net(image)
    # output = torch.atan2(output[0][0], output[0][1])
    # output = torch.arctan(output[0][0] / output[0][1])
    output = torch.atan2(output[0][1], output[0][0])
    output = output.item()
    return output

    # return output[0][0].item(), output[0][1].item()

file_path = args.data_path
csv_path = args.data_path + '/data.csv'

# add a new column to the csv file called new_prediction
image_info = pd.read_csv(csv_path)

# check whether the column 'ground_truth' exists

# if 'ground_truth' not in image_info.columns:
#     image_info['ground_truth'] = 0

image_info[f'{model}'] = 0
# image_info[f'{model}_pred1'] = 0
# image_info[f'{model}_pred2'] = 0

print('Predicting...')

for file in os.listdir(file_path):
    if file.endswith('.jpg'):
        img_path = os.path.join(file_path, file)
        pred = predict(img_path)
        # pred1, pred2 = predict(img_path)
        idx = image_info[image_info['path_idx'] == int(file.split('_')[0])].index[0]

        image_info.at[idx, f'{model}'] = pred
        # image_info.at[idx, f'{model}_pred1'] = pred1
        # image_info.at[idx, f'{model}_pred2'] = pred2

image_info.to_csv(csv_path, index=False)
print('Done')