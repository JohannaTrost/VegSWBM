# %%
# os.chdir('..')
# %%
# Imports
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from src.swbm import *
from src.plots import *
from src.utils import *

#%%
# Load and pre-process data
input_swbm_raw = pd.read_csv('data/Data_swbm_Germany.csv')
input_swbm = prepro(input_swbm_raw)

#%%
# Calibration (opt c_s, g, a, b0 seasonal)