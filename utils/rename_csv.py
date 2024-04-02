import pandas as pd
import numpy as np
data = pd.read_csv('/home/aoqiao/developer_dq/homingdrone-learning/data_gym/processed_031901/data.csv')

data_csv = pd.read_csv('/home/aoqiao/developer_dq/homingdrone-learning/data_gym/processed_label_031901.csv')

with open('/home/aoqiao/developer_dq/homingdrone-learning/data_gym/processed_label_test_031901.csv', 'w') as f:
    f.write('filename, gaze, pos_x, pos_y, label_1, label_2\n')

    for i in range(26):
        path_idx = data['path_idx'][i]
        filename = f'{path_idx}_rected.jpg'
        heading = data['heading'][i]
        gaze = int((heading*180/np.pi)%360) 
        if gaze == 0:
            gaze = 360  
        pos_x = data['pos_x'][i]
        pos_y = data['pos_y'][i]
        #from data_csv, get label_1 and label_2, where pos_x and pos_y are the same
        # print(data_csv[(data_csv[' pos_x'] == pos_x) & (data_csv[' pos_y'] == pos_y)],'gg')
        label_1 = data_csv[(data_csv[' pos_x'] == pos_x) & (data_csv[' pos_y'] == pos_y) & (data_csv[' gaze'] == gaze)][' label_1'].values[0]
        label_2 = data_csv[(data_csv[' pos_x'] == pos_x) & (data_csv[' pos_y'] == pos_y) & (data_csv[' gaze'] == gaze)][' label_2'].values[0]
        f.write(f'{filename},{gaze},{pos_x},{pos_y},{label_1},{label_2}\n')
