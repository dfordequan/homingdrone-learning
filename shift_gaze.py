# first import ./data_gym/test_nocase/0_rected.jpg
import cv2
import numpy as np

for i in range(4,31):
    img = cv2.imread(f'/home/aoqiao/developer_dq/homingdrone-learning/data_gym/processed_samedirection/{i}_rected.jpg', cv2.IMREAD_GRAYSCALE)

    img = np.concatenate((img[:,450:], img[:,:450]), axis=1)

    # display the image
    cv2.imwrite(f'/home/aoqiao/developer_dq/homingdrone-learning/data_gym/processed_samedirection/{i}_rected.jpg', img)