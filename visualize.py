from preprocess import get_home_direction, deg_to_unit_vector, get_relative_home_direction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--column', type=str)
parser.add_argument('--data_path', type=str)
args = parser.parse_args()

file_path = args.data_path + '/data.csv'
column = args.column


data = pd.read_csv(file_path)
 
 
 
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
 
offset = 0
 
 
for i in range(len(data)):
    pos_x = data['pos_x_mocap'][i]
    pos_y = data['pos_y_mocap'][i]
    heading = data['heading_mocap'][i]
    prediction = data[f'{column}'][i]
 
    gaze = np.rad2deg(heading) % 360
 
    home_direction = get_home_direction(pos_x, pos_y)
    home_vector = deg_to_unit_vector(home_direction)
    relative_home_direction = get_relative_home_direction(home_vector, gaze)
 
    angle = heading+prediction+offset
 
    # convert angle from radians to unit vector
 
    x = np.cos(angle)
    y = np.sin(angle)
 
    ax.quiver(pos_x, -pos_y, x, -y, angles='xy', scale_units='xy', scale=5, color='blue', width=0.005)
    ax.quiver(pos_x, -pos_y, home_vector[0], -home_vector[1], angles='xy', scale_units='xy', scale=5, color='red', width=0.005)
    # draw a star at the home positiion (0,0)
    ax.plot(0, 0, 'o', markersize=5)
    # add legend
    ax.legend(['Predict', 'Ground'])
 
# plt.show()
dataname = args.data_path.split('/')[-1]
os.makedirs(f'./plots/{column}', exist_ok=True)
plt.savefig(f'./plots/{column}/{dataname}_{column}.png')