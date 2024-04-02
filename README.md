
### data preparation
After the training data gathered (learning flight), copy the entire rectilinear folder (yyyymmdd_hhmm_rectilinear) that contains rectilinear images and data.csv file into data folder

### training
<code>python3 train.py --data_path yyyymmdd_hhmm_rectilinear (--no_preprocess)</code>


preprocessing.py will first be executed and generated dataset and labels in data folder

then the network will be trained and output in the models file

the training loss will be saved in the log folder

### testing
<code>python3 test.py --model_path --data_path</code>

Test it with any dataset that has rectilinear images and data.csv

After it's done a new column will be added to data.csv

### visualizing
<code>python3 visualize.py --data_path ./data/20240325_combined --column 
gazenet_20240319_01</code>

### utils
some useful functions to manipulate the dataset

1. co_register.py: co-register the OptiTrack and OpticFlow data
2. data_combine: combine multiple batches of dataset
3. image_annotate.py: visualize the network output and ground truth ont he rectilinear images
4. rename_csv.py: turn a data.csv into a label csv (useful to get the ground truth for test dataset)
5. shift_gaze.py: virtually rotate the images gaze direction
6. single_image_test.py: test with a single image (used in pi-camera) 

### playground
some experimental things
