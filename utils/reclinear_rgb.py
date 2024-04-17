import cv2
import numpy as np
import pickle
import os

file_location = "/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240403/20240403_095708"
dir_name = file_location.split("/")[-1]
output_location = f"./data/{dir_name}_rgb"

csv_location = file_location + "_rectilinear/data.csv"

def read_image(file_name):
    image_org = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # image_gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)

    return image_org

def apply_mask(image_org, mask):
    image_masked = cv2.bitwise_and(image_org, image_org, mask=mask)
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
    with open("./utils/mask.pkl", "rb") as f:
        mask_file = pickle.load(f)

    mask = mask_file['mask']
    x,y,r = mask_file['x'], mask_file['y'], mask_file['r']
    return mask, x, y, r


def rectilinear(file_name, output_location, file_id):
	mask, x, y, r = get_mask()
	
	print(file_name)
	image_gray = read_image(file_name)
	image_masked = apply_mask(image_gray, mask)
	image_polar = convert_image(image_masked, x, y, r)
	image_final = after_process(image_polar)
	image_rotate = rotate_image(image_final)
	file_idx = file_name.split('/')[-1].split('.')[0].split('_')[0]
	cv2.imwrite(output_location + f"{file_idx}" + f'_rgb_{file_id}' + ".jpg", image_rotate)

os.makedirs(output_location, exist_ok=True)

# copy the csv file
os.system(f"cp {csv_location} {output_location}")


for file in os.listdir(file_location):
    if not file.endswith('.jpg'):
        continue
    
    rectilinear(file_location + "/" + file, output_location + "/", file.split('_')[-2]+'_'+file.split('_')[-1].split('.')[0])
    



