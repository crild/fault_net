### --------- function for making feature images from non-random starts --------

# Import needed functions and modules
import keras
import os

from fault_net_func.data_cond import *
from fault_net_func.segy_files import *
from fault_net_func.feature_vis import *

# Set the RNG
np.random.seed(7)

# Define some parameters
keras_model = keras.models.load_model('F3/fault25epochs_tall_test.h5')
end_str = 'Mirror1_2_3_T'
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
im_idx = 1 # 3/4/9

start_im, y = generator.data_generation(im_idx)

# Check if we need to make a new image directory
if not os.path.exists('images/image'+str(im_idx)):
    os.makedirs('images/image'+str(im_idx))

save_or(start_im,name = 'images/image'+str(im_idx)+'/Original_im')


layer_name = 'conv_layer1'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer1_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer1_gray'+end_str)

layer_name = 'conv_layer2'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer2_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer2_gray'+end_str)

layer_name = 'conv_layer3'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer3_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer3_gray'+end_str)

layer_name = 'conv_layer4'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer4_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer4_gray'+end_str)

layer_name = 'conv_layer5'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer5_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer5_gray'+end_str)


layer_name = 'attribute_layer'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/attribute_layer_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/attribute_layer_gray'+end_str)


layer_name = 'pre-softmax_layer' #'attribute_layer'

(filter_list, losses1) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/pre-softmax_im'+end_str)

(filter_list, losses2) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/pre-softmax_gray'+end_str)



np.set_printoptions(precision=2)
print('Loss list1:')
print(losses1)
print('Loss list2:')
print(losses2)
