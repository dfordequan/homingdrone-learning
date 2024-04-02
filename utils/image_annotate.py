import os
import cv2
import pandas as pd
import numpy as np

file_path = '/home/aoqiao/developer_dq/homingdrone-learning/data_gym/train_031901/'
csv_path = '/home/aoqiao/developer_dq/homingdrone-learning/data_gym/processed_label_031901.csv'

if not os.path.exists(file_path[:-1]+ '_line'):
    os.makedirs(file_path[:-1]+ '_line')

df = pd.read_csv(csv_path)

for i in range(len(df)):
    filename = df['filename'][i]
    label_1 = df[' label_1'][i]
    label_2 = df[' label_2'][i]
    # calculate the angle in radians based on the label_1 and label_2, between -pi and pi
    angle = np.arctan2(label_2, label_1)
    print(angle)
    # read the image
    img = cv2.imread(file_path + filename)
    line_position = 900 + int(angle*900/np.pi)
    # draw a vertical line on the line position

    img = cv2.line(img, (line_position, 0), (line_position, 1800), (255, 0, 0), 2)
    cv2.imwrite(file_path[:-1]+ '_line/' + filename+'_line.jpg', img)