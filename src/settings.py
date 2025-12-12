import sys, os
import pandas as pd
import sys
import numpy as np
import random
import time
import pickle
import psutil
import gc


from IPython.display import Image, display

import warnings
warnings.filterwarnings("ignore")



from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 2000)




src = r"C:\Users\user\Desktop\Coding mo\AutoML proj1\task 1\AutoML Project Molham\src"
sys.path.append(src)

from paths import *

get_paths()




RANDOM_SEEDS = [42, 123]
N_FOLDS = 5

# TIME_BUDGET = 600 # too long
# CV_TIME_BUDGET = 600 # too long
TIME_BUDGET = 30

MEMORY_LIMIT = 4*1024


CV_TIME_BUDGET = 10

CV_MEMORY_LIMIT = 1*1024

TARGET_COLS = {
    "modeldata.csv": "IsInsurable",
    "salary.csv": "Salary",
    "train.csv": "variety",
    "titanic.csv": "Survived",
    "wine.csv": "Wine"
}


TEST_SIZE = 0.2
RANDOM_STATE = RANDOM_SEEDS[0]
SCORING_METRICS = ["accuracy", "f1", "precision", "recall"]




notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, "../.."))
sys.path.append(os.path.join(project_root, "src"))



