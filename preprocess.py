import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm

def center_gaze_direction(image, degree, gaze_org):

    cols_per_degree = image.shape[1] // 360

    north_col = - cols_per_degree * gaze_org + image.shape[1]//2
    
    center_col = degree * cols_per_degree + north_col

    if center_col >= image.shape[1]:
        center_col -= image.shape[1]

    left_col = center_col - image.shape[1]//2

    output_image = np.concatenate((image[:, left_col:], image[:, :left_col]), axis=1)

    return output_image


def get_home_direction(pos_x, pos_y):

    if pos_x<0:
        return np.rad2deg(np.arctan(pos_y/pos_x))
    else:
        return np.rad2deg(min(np.pi + np.arctan(pos_y/pos_x), -np.pi + np.arctan(pos_y/pos_x), key=abs))
    

def deg_to_unit_vector(deg):

    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])


def get_relative_home_direction(home_direction, gaze):

    gaze_matrix = np.array([[np.cos(np.deg2rad(gaze)), np.sin(np.deg2rad(gaze))], [-np.sin(np.deg2rad(gaze)), np.cos(np.deg2rad(gaze))]])

    relative_home_direction = np.matmul(gaze_matrix, home_direction)

    return relative_home_direction


def preprocess(rectilinear_path):
    data_name = rectilinear_path.split('/')[-1]

    data_csv = f'{rectilinear_path}/data.csv'

    image_info = pd.read_csv(data_csv)  

    output_csv = f'{rectilinear_path}/label_training_{data_name}.csv'
    output_location = f'{rectilinear_path}/training_{data_name}/'

    if not os.path.exists(output_location):
        os.makedirs(output_location)

    with open(output_csv, 'w') as f:
        f.write('filename, gaze, pos_x, pos_y, label_1, label_2\n')
        for file in tqdm(os.listdir(rectilinear_path)):
            if not file.endswith('.jpg'):
                continue
            img = cv2.imread(os.path.join(rectilinear_path, file))
            
            i = int(file.split('_')[0])

            #from image_info, find the row that path_idx == i
            pos_x = image_info[image_info['path_idx'] == i]['pos_x'].values[0]
            pos_y = image_info[image_info['path_idx'] == i]['pos_y'].values[0]
            heading_org = image_info[image_info['path_idx'] == i]['heading'].values[0]
            gaze_org = (heading_org*180/np.pi)%360

            gaze_org = int(gaze_org)


            home_angle_deg = get_home_direction(pos_x,pos_y)

            home_vector = deg_to_unit_vector(home_angle_deg)

            for k in range(1, 361):
                gaze = k

                if gaze > 360:
                   gaze -= 360
                elif gaze < 1:
                    gaze += 360
            
                image_gaze = center_gaze_direction(img, gaze, gaze_org)

            
                label = get_relative_home_direction(home_vector, gaze)


                if gaze < 10:
                    gaze = "00" + str(gaze)
                elif gaze < 100:
                    gaze = "0" + str(gaze)
                else:
                    gaze = str(gaze)

                filename_gaze = str(i) + "_" + gaze + ".jpg"
                cv2.imwrite(os.path.join(output_location, filename_gaze), image_gaze)

                f.write(filename_gaze + "," + str(gaze) + "," + str(pos_x) + "," + str(pos_y) + "," + str(label[0]) + "," + str(label[1]) + "\n")


