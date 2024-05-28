from preprocess import get_home_direction, deg_to_unit_vector, get_relative_home_direction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--column', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--mocap', action='store_true', dest='mocap')
args = parser.parse_args()

file_path = args.data_path + '/data.csv'
column = args.column


data = pd.read_csv(file_path)
 


def get_angular_error(prediction, ground_truth):
    error = prediction - ground_truth
    error = np.rad2deg(error)
    # make sure the error is between -180 and 180
    if error > 180:
        error -= 360
    elif error < -180:
        error += 360
    return error

fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
 

offset = 0

data[f'{args.column}_error'] = 0
 
for i in range(len(data)):
    if args.mocap:
        pos_x = data['pos_x_mocap'][i]
        pos_y = data['pos_y_mocap'][i]
        heading = data['heading_mocap'][i]
    else:
        pos_x = data['pos_x'][i]
        pos_y = data['pos_y'][i]
        heading = data['heading'][i]
    prediction = data[f'{column}'][i]
    confidence = data[f'{column}_conf'][i]
 
    gaze = np.rad2deg(heading) % 360
 
    home_direction = get_home_direction(pos_x, pos_y)
    home_vector = deg_to_unit_vector(home_direction)
    relative_home_direction = get_relative_home_direction(home_vector, gaze)
 
    angle = heading+prediction+offset

    angle_gt = heading+offset+np.arctan2(relative_home_direction[1], relative_home_direction[0])

    error = get_angular_error(angle, angle_gt)
    print(f'Error: {error}')
    data[f'{args.column}_error'][i] = error
 
    # convert angle from radians to unit vector
 
    x = np.cos(angle)
    y = np.sin(angle)
 
    ax.quiver(pos_x, -pos_y, home_vector[0], -home_vector[1], angles='xy', scale_units='xy', scale=5, color='gray', width=0.005)
    ax.quiver(pos_x, -pos_y, x, -y, angles='xy', scale_units='xy' , scale=5, color='#A64294', width=0.005)
    
    # draw a star at the home positiion (0,0)
    ax.plot(0, 0, 'o', markersize=5)
    # add legend
    ax.legend(['Ground', 'Prediction'])
    ax.set_title('Vector output from the model')
 
# plt.show()
dataname = args.data_path.split('/')[-1]
os.makedirs(f'./plots/{column}/{dataname}', exist_ok=True)
plt.savefig(f'./plots/{column}/{dataname}/{dataname}_{column}_prediction.png')

# Confidence Plot
# Define the colors
colors = ["#E3EFFB", "#5C8DC7"]
cmap_name = 'confidence_cmap'

# Create the colormap
conf_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

def custom_cmap(value, cmap, norm, low_val_color=(1.0, 0.0, 0.0, 1.0), threshold=0.05):
    if value < threshold:
        print('Low confidence')
        return low_val_color
    else:
        return cmap(norm(value))

fig1, ax1 = plt.subplots()
norm_conf = Normalize(vmin=data[f'{args.column}_conf'].min(), vmax=data[f'{args.column}_conf'].max())

# Apply custom colormap
confidence_values = data[f'{args.column}_conf']
colors = [custom_cmap(val, conf_cmap, norm_conf) for val in confidence_values]

# if args.mocap:
#     sc1 = ax1.scatter(data['pos_x_mocap'], -data['pos_y_mocap'], c=data[f'{args.column}_conf'], cmap=conf_cmap, norm=norm_conf)
# else:
#     sc1 = ax1.scatter(data['pos_x'], -data['pos_y'], c=data[f'{args.column}_conf'], cmap=conf_cmap, norm=norm_conf)

if args.mocap:
    sc1 = ax1.scatter(data['pos_x_mocap'], -data['pos_y_mocap'], c=colors)
else:
    sc1 = ax1.scatter(data['pos_x'], -data['pos_y'], c=colors)
# Adding a colorbar with the custom colormap
sm = plt.cm.ScalarMappable(cmap=conf_cmap, norm=norm_conf)
sm.set_array([])
fig1.colorbar(sm, ax=ax1, orientation='vertical')
ax1.set_title('Distance map')
ax1.set_xlabel('Position X (mocap)')
ax1.set_ylabel('Position Y (mocap)')
ax1.grid(True)
plt.savefig(f'./plots/{column}/{dataname}/{dataname}_{column}_confidence.png')
plt.close(fig1)

# Angular Error Plot

# Define the colors
colors = ["#FBEBDE", "#EF7F29"]  # light pink to orange
cmap_name = 'error_cmap'

# Create the colormap
error_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

fig2, ax2 = plt.subplots()
if args.mocap:
    sc2 = ax2.scatter(data['pos_x_mocap'], -data['pos_y_mocap'], c=abs(data[f'{args.column}_error']), cmap=error_cmap)
else:
    sc2 = ax2.scatter(data['pos_x'], -data['pos_y'], c=abs(data[f'{args.column}_error']), cmap=error_cmap)
fig2.colorbar(sc2, ax=ax2, orientation='vertical')
ax2.set_title('Angular Errors')
ax2.set_xlabel('Position X (mocap)')
ax2.set_ylabel('Position Y (mocap)')
ax2.grid(True)
plt.savefig(f'./plots/{column}/{dataname}/{dataname}_{column}_angular_error.png')
plt.close(fig2)

# make a histogram of the errors, including the mean and std, print them on the plot
fig3, ax3 = plt.subplots()
ax3.hist(data[f'{args.column}_error'], bins=20)
ax3.set_title('Angular Error Histogram')
ax3.set_xlabel('Error (degrees)')
ax3.set_ylabel('Frequency')
mean_error = np.mean(data[f'{args.column}_error'])
std_error = np.std(data[f'{args.column}_error'])
ax3.axvline(mean_error, color='k', linestyle='dashed', linewidth=1)
ax3.axvline(mean_error + std_error, color='r', linestyle='dashed', linewidth=1)
ax3.axvline(mean_error - std_error, color='r', linestyle='dashed', linewidth=1)
ax3.legend([f'Mean={round(mean_error,2)}', f'Mean+Std({round(std_error,2)})', f'Mean-Std({round(std_error,2)})'])
plt.savefig(f'./plots/{column}/{dataname}/{dataname}_{column}_angular_error_hist.png')
plt.close(fig3)

print(f'Mean Error: {mean_error}')
print(f'Std Error: {std_error}')