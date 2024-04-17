# read a csv file that has a column 'Iteration' and 'Loss' and plot the loss
import pandas as pd
import matplotlib.pyplot as plt

# open the csv file

file = '/home/aoqiao/developer_dq/homingdrone-learning/logs/20240403_090245.csv'

# read the csv file

data = pd.read_csv(file)

# get the column 'Iteration' and 'Loss'

iteration = data['Iteration']
loss = data['Loss']

# plot the loss

plt.plot(iteration, loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')

plt.show()