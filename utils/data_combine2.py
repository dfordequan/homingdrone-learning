import os
import cv2
import numpy as np
import pandas as pd


dataset1 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_180124/'
dataset2 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_181319/'
dataset3 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_182751/'
dataset4 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_183641/'

csv1 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_180124/data.csv'
csv2 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_181319/data.csv'
csv3 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_182751/data.csv'
csv4 = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_183641/data.csv'

save_path = '/home/aoqiao/developer_dq/homingdrone-learning/data/20240325_combined'

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open (save_path + '/' + 'data.csv', 'w') as f:
    f.write('path_idx,pos_x,pos_y,heading,prediction\n')
    for dataset, csv in zip([dataset1, dataset2, dataset3, dataset4], [csv1, csv2, csv3, csv4]):
        image_info = pd.read_csv(csv)
        sfx = int(dataset.split('_')[-1][:6])
        for file in os.listdir(dataset):
            if not file.endswith('.jpg'):
                continue
            img = cv2.imread(dataset + file, cv2.IMREAD_GRAYSCALE)
            i = int(file.split('_')[0])
            pos_x = image_info[image_info['path_idx'] == i]['pos_x'].values[0]
            pos_y = image_info[image_info['path_idx'] == i]['pos_y'].values[0]
            heading = image_info[image_info['path_idx'] == i]['heading'].values[0]
            prediction = image_info[image_info['path_idx'] == i]['prediction'].values[0]
            new_i = i + sfx
            new_file = f'{new_i}_{file}'
            cv2.imwrite(save_path + '/' + new_file, img)
            f.write(f'{new_i},{pos_x},{pos_y},{heading},{prediction}\n')

            

