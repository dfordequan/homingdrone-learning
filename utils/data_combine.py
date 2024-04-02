import os
import cv2
import numpy as np
import pandas as pd



data_home = './data_gym/test'
data_nextpoint = './data_gym/test_nextpoint'
data_samedirection = './data_gym/processed_samedirection'

csv_home = pd.read_csv('./data_gym/processed_label_test.csv')
csv_nextpoint = pd.read_csv('./data_gym/processed_label_test_nextpoint.csv')
csv_samedirection = pd.read_csv('./data_gym/processed_label_test_samedirection.csv')

save_path = './data_gym/test_combined'

# combine all the png files from each folder, add a suffix to the filename, save to a new folder
# find their corresponding csv files, alter the name, combine them, save to a new csv file


with open (save_path + '/' + 'processed_label_test_combined.csv', 'w') as f:

    f.write('filename,gaze,pos_x,pos_y,label_1,label_2\n')
    for filename in os.listdir(data_home):
        if filename.endswith('.jpg'):
            img = cv2.imread(data_home + '/' + filename)
            cv2.imwrite(save_path + '/' + 'home_' + filename, img)
            print('home', filename)
            # find the corresponding csv file, alter the name, combine them, save to a new csv file
            gaze = int(csv_home[csv_home['filename'] == filename][' gaze'].values[0])
            pos_x = csv_home[csv_home['filename'] == filename][' pos_x'].values[0]
            pos_y = csv_home[csv_home['filename'] == filename][' pos_y'].values[0]
            label_1 = csv_home[csv_home['filename'] == filename][' label_1'].values[0]
            label_2 = csv_home[csv_home['filename'] == filename][' label_2'].values[0]
            f.write(f'home_{filename},{gaze},{pos_x},{pos_y},{label_1},{label_2}\n')

    for filename in os.listdir(data_nextpoint):
        if filename.endswith('.jpg'):
            img = cv2.imread(data_nextpoint + '/' + filename)
            cv2.imwrite(save_path + '/' + 'nextpoint_' + filename, img)
            print('nextpoint', filename)
            # find the corresponding csv file, alter the name, combine them, save to a new csv file
            gaze = int(csv_nextpoint[csv_nextpoint['filename'] == filename][' gaze'].values[0])
            pos_x = csv_nextpoint[csv_nextpoint['filename'] == filename][' pos_x'].values[0]
            pos_y = csv_nextpoint[csv_nextpoint['filename'] == filename][' pos_y'].values[0]
            label_1 = csv_nextpoint[csv_nextpoint['filename'] == filename][' label_1'].values[0]
            label_2 = csv_nextpoint[csv_nextpoint['filename'] == filename][' label_2'].values[0]
            f.write(f'nextpoint_{filename},{gaze},{pos_x},{pos_y},{label_1},{label_2}\n')

    for filename in os.listdir(data_samedirection):
        if filename.endswith('.jpg'):
            img = cv2.imread(data_samedirection + '/' + filename)
            cv2.imwrite(save_path + '/' + 'samedirection_' + filename, img)
            print('samedirection', filename)
            # find the corresponding csv file, alter the name, combine them, save to a new csv file
            gaze = int(csv_samedirection[csv_samedirection['filename'] == filename][' gaze'].values[0])
            pos_x = csv_samedirection[csv_samedirection['filename'] == filename][' pos_x'].values[0]
            pos_y = csv_samedirection[csv_samedirection['filename'] == filename][' pos_y'].values[0]
            label_1 = csv_samedirection[csv_samedirection['filename'] == filename][' label_1'].values[0]
            label_2 = csv_samedirection[csv_samedirection['filename'] == filename][' label_2'].values[0]
            f.write(f'samedirection_{filename},{gaze},{pos_x},{pos_y},{label_1},{label_2}\n')

print('done')
        