# fault_net
Fault detection software for seismic data

Functions needed to run fault_net.py are contained in the folder fault_net_func, logs hold TensorBoard data, and F3 holds trained models.

The user needs the F3 dataset in the same folder as facies_net.py, as well as classification in .pts files.

The file will save training results in TensorBoard-format, simply go into the terminal and write: tensorboard --logdir=logs/"name-of-folder" and then open a new web browser and write "localhost:6006" in the adress bar.

e.g. to view the results from F3_train write: tensorboard --logdir=logs/F3 then open a new tab in chrome and input localhost:6006

Optimized for TensorFlow 1.5.0
