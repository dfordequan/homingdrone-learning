import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale
import os
import pandas as pd
from models.model import CompactCNN
from models.model_rgb import CompactCNN_rgb
from models.model_ds import CompactCNN_ds
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--rgb', action='store_true', dest='rgb')
parser.add_argument('--ds', action='store_true', dest='ds')



args = parser.parse_args()

if args.rgb:
    net = CompactCNN_rgb()
    suffix = '_rgb'
    # define the transforms
    transform = transforms.Compose([
    transforms.Resize((192, 1800)),
    transforms.ToTensor()
    ])
elif args.ds:
    net = CompactCNN_ds()
    suffix = '_ds'
    # define the transforms
    transform = transforms.Compose([
    transforms.Resize((48, 450)),
    transforms.ToTensor()
    ])
else:
    net = CompactCNN()
    size = (192, 1800)
    suffix = ''
    # define the transforms
    transform = transforms.Compose([
    transforms.Resize((192, 1800)),
    Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])
    
model_path = args.model_path

model = model_path.split('/')[-1][:-4]



net.load_state_dict(torch.load(model_path))
    

def predict(img_path, ds=False):
    
    image = Image.open(img_path)

    if ds:
        image = image.resize((450, 48))


    image = transform(image)
    image = image.unsqueeze(0)

    output = net(image)
    # output = torch.atan2(output[0][0], output[0][1])
    # output = torch.arctan(output[0][0] / output[0][1])
    prediction= torch.atan2(output[0][1], output[0][0])
    prediction = prediction.item()
    # compute the norm of the output, interpret it as the confidence, which is abs of 1 - norm
    norm = torch.norm(output)
    confidence = abs(1 - norm.item())
    ### for distance, which is the magnitude of the output
    confidence = norm.item()
    return prediction, confidence

    # return output[0][0].item(), output[0][1].item()

file_path = args.data_path
csv_path = args.data_path + '/data.csv'

# add a new column to the csv file called new_prediction
image_info = pd.read_csv(csv_path)

# check whether the column 'ground_truth' exists

# if 'ground_truth' not in image_info.columns:
#     image_info['ground_truth'] = 0

image_info[f'{model}'] = 0
image_info[f'{model}_conf'] = 0
# image_info[f'{model}_pred1'] = 0
# image_info[f'{model}_pred2'] = 0

print('Predicting...')

for file in os.listdir(file_path):
    if file.endswith('.jpg'):
        img_path = os.path.join(file_path, file)
        pred, conf = predict(img_path, args.ds)
        # pred1, pred2 = predict(img_path)
        i = file.split('_')[0]

        date = file.split('_')[2]
        time = file.split('_')[3].split('.')[0]

        i = i + "_" + date + "_" + time


        idx = image_info[image_info['path_idx'] == i].index[0]

        image_info.at[idx, f'{model}'] = pred
        image_info.at[idx, f'{model}_conf'] = conf
        # image_info.at[idx, f'{model}_pred1'] = pred1
        # image_info.at[idx, f'{model}_pred2'] = pred2

image_info.to_csv(csv_path, index=False)
print('Done')

