# constants.py
# Applicable constants for HW2

import numpy as np
import os

home_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = home_dir_path + '/data/'
results_path = home_dir_path + '/results/'

png_exten = '.png'
txt_exten = '.txt'

reg_lambda = 1E-1

input_variables = ["agePct12t29", "pctWSocSec",
                   "pctUrban", "agePct65up", "householdsize"]

one_to_digit = { -1 : 2, +1 : 7}
digit_to_one = { 2 : -1, 7 : +1}

mnist_step_size = 1E-1
cutoff = 1E-2

stoch_iter_count = 150
