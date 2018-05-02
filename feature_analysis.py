### --------- function for making feature images from non-random starts --------

# Import needed functions and modules
import keras

from fault_net_func.data_cond import *
from fault_net_func.segy_files import *
from fault_net_func.feature_vis import *


# Define some parameters
keras_model = keras.models.load_model('F3/fault.h5')
layer_name = 'pre-softmax_layer' #'conv_layer2' #'attribute_layer'
cube_incr = 30
segy_filename = ['F3_entire.segy']
file_list = ['./class_addresses/multi_else_ilxl.pts','./class_addresses/f3_fault_training_locations.pts'] # list of names of class-adresses

# Store all the segy-data and specifications as a segy object
segy_obj = segy_reader(segy_filename)

print('Making class-adresses')
tr_adr,val_adr = convert(file_list = file_list,
                         save = False,
                         savename = None,
                         ex_adjust = True,
                         val_split = 0.3)
print('Finished making class-adresses')

# Define parameters for the generators
tr_params =        {'seis_spec'   : segy_obj,
                    'adr_list'    : tr_adr,
                    'cube_incr'   : cube_incr,
                    'num_classes' : len(file_list),
                    'batch_size'  : 1,
                    'steps'       : 100,
                    'print_info'  : True}

generator = ex_create(**tr_params)

# image index fram tr_adr (must beless than steps in tr_params)
im_idx = 50

start_im, y = generator.data_generation(im_idx)

save_or(start_im,name = 'images/Original_im')

(filter_list, losses1) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/pre-softmax_im_start')

(filter_list, losses2) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/pre-softmax_im_gray')

np.set_printoptions(precision=2)
print('Loss list1:')
print(losses1)
print('Loss list2:')
print(losses2)
