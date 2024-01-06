import cv2
import numpy as np

file_location = "./data_gym/processed/"


def center_gaze_direction(image, degree, north_col):

    """
    degree: The gaze direction in degrees. Must be between 1 and 360.
    north_col: The col number in the image that corresponds to the north direction.
    """

    degree  = degree - 180
    if not 1 <= degree <= 360:
        degree+=360

    if not 0 <= north_col < image.shape[1]:
        raise ValueError("North col must be between 0 and the number of cols in the image")

    cols_per_degree = image.shape[1] // 360
    center_col = degree * cols_per_degree + north_col

    if center_col >= image.shape[1]:
        center_col -= image.shape[1]

    if center_col < image.shape[1]/2:
        output_image = np.concatenate((image[:, center_col:], image[:, :center_col]), axis=1)
    else:
        output_image = np.concatenate((image[:, center_col:], image[:, :center_col]), axis=1)

    return output_image

def get_home_direction(pos_x, pos_y):
    """
    input the x,y coordinate with nest at (0,0) and y towards north

    returns the angle in degrees that the drone have to turn from facing y to origin in clockwise direction
    
    """

    angle = np.arccos(-pos_y/np.sqrt(pos_x**2+pos_y**2))

    # if right half plane, convert to degrees and subtract 180
    if pos_x > 0:
        return -np.rad2deg(angle)
    
    # if left half plane, keep it as it is and convert to degrees
    else:
        return np.rad2deg(angle)
    

def deg_to_unit_vector(deg):
    """
    input the angle in degrees

    returns the unit vector in the direction of the angle
    """
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])


def get_relative_home_direction(home_direction, gaze):
    gaze_matrix = np.array([[np.cos(np.deg2rad(gaze)), -np.sin(np.deg2rad(gaze))], [np.sin(np.deg2rad(gaze)), np.cos(np.deg2rad(gaze))]])
    relative_home_direction = np.matmul(gaze_matrix, home_direction)

    return relative_home_direction


# save data to a csv file
output_location = "./data_gym/training/"
with open("./data_gym/processed_label.csv", "a") as f:
    f.write("filename, id, gaze, pos_x, pos_y, label_1, label_2\n")
    for i in range(1,12):
        for j in range(1,12):
            row = chr(i+64)
            col = j
            if col < 10:
                col = "0" + str(col)
            else:
                col = str(col)
            id = str(row) + str(col)
            filename = 'gym_' + id
            image = cv2.imread(file_location + filename + ".jpg", cv2.IMREAD_GRAYSCALE)

            #determine the pos_x and pos_y, the grid is 11(A-K) x 11(1-11), the center is (6,6)
            pos_x = 6 - i
            pos_y = j - 6
            # skip (0,0)
            if pos_x == 0 and pos_y == 0:
                continue
            
            home_angle_deg = get_home_direction(pos_x,pos_y)

            home_vector = deg_to_unit_vector(home_angle_deg)

            


    
            for k in range(1, 361):
                # image_gaze = center_gaze_direction(image, k, 900)

                gaze = k
                label = get_relative_home_direction(home_vector, gaze)

                # padding k to 3 digits
                if k < 10:
                    k = "00" + str(k)
                elif k < 100:
                    k = "0" + str(k)
                else:
                    k = str(k)

                filename_k = filename + "_" + k + ".jpg"
                cv2.imwrite(output_location + filename_k, image_gaze)
    
                f.write(filename_k + "," + id + "," + str(gaze) + "," + str(pos_x) + "," + str(pos_y) + "," + str(label[0]) + "," + str(label[1]) + "\n")