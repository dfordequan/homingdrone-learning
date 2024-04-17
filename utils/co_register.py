import numpy as np
import os
import pandas as pd

optitrack_file = '/home/aoqiao/developer_dq/thesis_data/data_cyberzoo/20240403/20240403/1151.csv'
up = 'y' # or 'y'
time_offset = -72.515
# x_offset = 0.0
# y_offset = 0.0
# z_offset = 0.0


ot_data = pd.read_csv(optitrack_file)

# remove the first row
ot_data = ot_data[1:]



ot_data['x[m]_revised'] = 0
ot_data['y[m]_revised'] = 0
ot_data['z[m]_revised'] = 0
ot_data['time[s]_revised'] = 0

for i in range(1,len(ot_data)):
    x = ot_data['x[m]'][i]
    y = ot_data['y[m]'][i]
    z = ot_data['z[m]'][i]
    t = ot_data['timestamp[us]'][i]

    if up == 'y':

        x_revised = z
        y_revised = -x
        z_revised = -y

    elif up == 'z':
        x_revised = -y
        y_revised = -x
        z_revised = -z
    t_revised = t/1000000

    if i == 1:
        time_offset_temp = t_revised + time_offset
        t_revised = -time_offset
        time_offset = time_offset_temp
        
        
    else:
        t_revised -= time_offset




    ot_data['x[m]_revised'][i] = x_revised
    ot_data['y[m]_revised'][i] = y_revised
    ot_data['z[m]_revised'][i] = z_revised
    ot_data['time[s]_revised'][i] = t_revised

ot_data.to_csv(optitrack_file[:-4] + '_revised.csv', index=False)