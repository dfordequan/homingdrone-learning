# first import ./data_gym/test_nocase/0_rected.jpg
import cv2
import numpy as np
import os

file_path = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240403/20240403_095708_rectilinear/'

save_path = file_path + 'shifted_10/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in os.listdir(file_path):
    if not file.endswith('.jpg'):
        continue
    
    img = cv2.imread(file_path + file)
    print(file)
    img = np.concatenate((img[:,50:], img[:,:50]), axis=1)

   
    cv2.imwrite(f'{save_path}{file}', img)