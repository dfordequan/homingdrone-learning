import cv2
import numpy as np
import os
import pandas as pd

file_location = "./data_gym/processed_nextpoint_landmarke/"

output_location = "./data_gym/train_nextpoint_landmarke_case/"
os.makedirs(output_location, exist_ok=True)
output_csv = "./data_gym/processed_label_nextpoint_landmarke_case.csv"


def center_gaze_direction(image, degree, gaze_org):

    """
    degree: The gaze direction in degrees. Must be between 1 and 360.
    north_col: The col number in the image that corresponds to the north direction.
    """

    cols_per_degree = image.shape[1] // 360

    north_col = - cols_per_degree * gaze_org + image.shape[1]//2

    # degree  = degree - 180
    # if not 1 <= degree <= 360:
    #     degree+=360

    
    center_col = degree * cols_per_degree + north_col

    if center_col >= image.shape[1]:
        center_col -= image.shape[1]

    left_col = center_col - image.shape[1]//2

    output_image = np.concatenate((image[:, left_col:], image[:, :left_col]), axis=1)

    ##########only take the upper half of the image
    output_image = output_image[:int(output_image.shape[0]), :]
    ##############################################


    return output_image

def get_home_direction(pos_x, pos_y):
    """
    # input the x,y coordinate with nest at (0,0) and y towards north

    # returns the angle in degrees that the drone have to turn from facing y to origin in clockwise direction
    
    # """

    # angle = np.arccos(-pos_y/np.sqrt(pos_x**2+pos_y**2))

    # # if right half plane, convert to degrees and subtract 1`8`0
    # if pos_x > 0:
    #     return -np.rad2deg(angle)
    
    # # if left half plane, keep it as it is and convert to degrees
    # else:
    #     return np.rad2deg(angle)

    if pos_x<0:
        return np.rad2deg(np.arctan(pos_y/pos_x))
    else:
        return np.rad2deg(min(np.pi + np.arctan(pos_y/pos_x), -np.pi + np.arctan(pos_y/pos_x), key=abs))

    
    

def deg_to_unit_vector(deg):
    """
    input the angle in degrees

    returns the unit vector in the direction of the angle
    """
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])


def get_relative_home_direction(home_direction, gaze, pos_x):
    # if pos_x > 0:
    gaze_matrix = np.array([[np.cos(np.deg2rad(gaze)), np.sin(np.deg2rad(gaze))], [-np.sin(np.deg2rad(gaze)), np.cos(np.deg2rad(gaze))]])
    # elif pos_x <= 0:
    # gaze_matrix = np.array([[np.cos(np.deg2rad(gaze)), -np.sin(np.deg2rad(gaze))], [np.sin(np.deg2rad(gaze)), np.cos(np.deg2rad(gaze))]])
    relative_home_direction = np.matmul(gaze_matrix, home_direction)

    return relative_home_direction

def regenerate_the_path(n, m):
    a = 0
    b = 0.1

    theta = np.linspace(0, 2*n*np.pi, m)

    r = a + b*theta

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    path = []
    for i in range(m):
        path.append((float(x[i]), float(y[i])))


    return path


# path = regenerate_the_path(4, 30)

# load the csv data 
path = []
image_info = pd.read_csv(file_location + "data.csv")

# save data to a csv file

with open(output_csv, "w") as f:
    f.write("filename, gaze, pos_x, pos_y, label_1, label_2\n")
    for i in range(3, 31):
        for filename in os.listdir(file_location):
            if filename.startswith(f"{i}_") and filename.endswith(".jpg"):
                break
        image = cv2.imread(file_location + filename, cv2.IMREAD_GRAYSCALE)

        #from image_info, find the row that path_idx == i
        pos_x = image_info[image_info['path_idx'] == i]['pos_x'].values[0]
        pos_y = image_info[image_info['path_idx'] == i]['pos_y'].values[0]
        heading_org = image_info[image_info['path_idx'] == i]['heading'].values[0]
        gaze_org = (heading_org*180/np.pi)%360

        # round gaze to no decimal
        gaze_org = int(gaze_org)

        # skip (0,0)
        if pos_x == 0 and pos_y == 0:
            continue

    
        home_angle_deg = get_home_direction(pos_x,pos_y)

        home_vector = deg_to_unit_vector(home_angle_deg)

        # print(file_name)



        for k in range(1, 361):
            #the range of gaze should be between 1 and 360, inclusive, gaze = gaze_org + k
            # so when gaze_org + k > 360, minus 360
            # when gaze_org + k < 1, add 360

            # gaze = gaze_org + k
            gaze = k

            if gaze > 360:
                gaze -= 360
            elif gaze < 1:
                gaze += 360
            
            image_gaze = center_gaze_direction(image, gaze, gaze_org)

            
            label = get_relative_home_direction(home_vector, gaze, pos_x)

            
            # gaze = gaze if gaze <= 360 else gaze - 360

            # padding k to 3 digits
            if gaze < 10:
                gaze = "00" + str(gaze)
            elif gaze < 100:
                gaze = "0" + str(gaze)
            else:
                gaze = str(gaze)

            filename_gaze = filename[:-4] + "_" + gaze + ".jpg"
            cv2.imwrite(output_location + filename_gaze, image_gaze)

            f.write(filename_gaze + "," + str(gaze) + "," + str(pos_x) + "," + str(pos_y) + "," + str(label[0]) + "," + str(label[1]) + "\n")