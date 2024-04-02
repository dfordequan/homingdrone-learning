# first import ./data_gym/test_nocase/0_rected.jpg
import cv2
import numpy as np
import os

file_path = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240325/20240325_183641/'

save_path = file_path + 'shifted_180/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in os.listdir(file_path):
    if not file.endswith('.jpg'):
        continue
    
    img = cv2.imread(file_path + file)
    print(file)
    img = np.concatenate((img[:,900:], img[:,:900]), axis=1)

   
    cv2.imwrite(f'{save_path}{file}', img)