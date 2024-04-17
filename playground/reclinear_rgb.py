import cv2
import numpy as np
import pickle
import os

file_location = "/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240311/20240311_180839_landmarkc"

def read_image(file_name):
    image_org = cv2.imread(file_name, cv2.IMREAD_COLOR)

    return image_org

def apply_mask(image_grgb, mask):
    image_masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return image_masked

def convert_image(image_masked, x, y, r):
    image_polar = cv2.linearPolar(image_masked, (x, y), r, cv2.WARP_FILL_OUTLIERS)

    return image_polar

def after_process(image_polar):
    image_rectilinear = cv2.rotate(image_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_flipped = np.flipud(image_rectilinear)
    # image_reshaped = cv2.resize(image_flipped[840:1450,:], (1800, 192))
    image_reshaped = cv2.resize(image_flipped[840:,:], (1800, 192))
    image_final = np.fliplr(image_reshaped)

    image_final = cv2.resize(image_final[7:175, :], (1800,192))

    return image_final

def get_gaze_angle(timestamp):
    # get gaze angle from the timestamp
    # return the gaze angle
    pass

def rotate_image(image_final):
    image_rotate = np.concatenate((image_final[:,450:], image_final[:,:450]), axis=1)
    return image_rotate


# read from pickle file
def get_mask():
    with open("./mask.pkl", "rb") as f:
        mask_file = pickle.load(f)

    mask = mask_file['mask']
    x,y,r = mask_file['x'], mask_file['y'], mask_file['r']
    return mask, x, y, r


mask, x, y, r = get_mask()

output_location = '../data/20240311_rgb/'

os.makedirs(output_location, exist_ok=True)


for i in range(len(os.listdir(file_location))):
    for filename in os.listdir(file_location):
        if filename.startswith(f"{i}") and filename.endswith(".jpg"):
        # if filename == f"{i}" + '.jpg':
            file_name = file_location + "/" + filename
            break
    print(file_name)
    image_rgb = read_image(file_name)
    image_masked = apply_mask(image_rgb, mask)
    image_polar = convert_image(image_masked, x, y, r)
    image_final = after_process(image_polar)
    image_rotate = rotate_image(image_final)
    cv2.imwrite(output_location + f"{i}_" + 'rgb' + ".jpg", image_rotate)


    



    
