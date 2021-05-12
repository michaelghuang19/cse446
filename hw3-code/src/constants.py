# constants.py
# Applicable constants for HW3

import numpy as np
import os

home_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = home_dir_path + '/data/'
results_path = home_dir_path + '/results/'

# kinda dumb, but arbitrary limit
hp_list = list(np.linspace(1, 28, 10))
lamb_list = [10 ** -x for x in np.linspace(1, 10, 10)]
