import pandas as pd
import os

import matplotlib.pyplot as plt
import numpy as np

csv1 = '/home/aoqiao/developer_dq/homingdrone-learning/data/20240403_095708_rectilinear/data.csv'
csv2 = '/home/aoqiao/developer_dq/homingdrone-learning/data/20240403_095708_shifted_10/data.csv'


# open csv1, for each row, get the column 'gazenet_', compare it with the same row in csv2 by store each of them in a list

image_info1 = pd.read_csv(csv1)
image_info2 = pd.read_csv(csv2)

gaze1 = np.rad2deg(image_info1['gazenet_'].values)
gaze2 = np.rad2deg(image_info2['gazenet_'].values)

# convert gaze1 from -180 to 180 to 0 to 360
gaze1 = (gaze1 + 360) % 360
gaze2 = (gaze2 + 360 +10) % 360


plt.plot(gaze1, label='gazenet_org')
plt.plot(gaze2, label='gazenet_10')

plt.legend()

plt.show()

