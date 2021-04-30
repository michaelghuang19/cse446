# Applicable constants for HW2

import numpy as np
import os

# from scipy import

home_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = home_dir_path + '/data/'
results_path = home_dir_path + '/results/'

png_exten = '.png'
txt_exten = '.txt'

reg_lambda = 10E-1



one_to_digit = { -1 : 2, +1 : 7}
digit_to_one = { 2 : -1, 7 : +1}

mnist_step_size = 10E-1
mnist_offset = 0
