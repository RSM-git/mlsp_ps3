from scipy.io import loadmat

import numpy as np

data = loadmat("problem3_6.mat")


t = data["t"].flatten()
y = data["y"].flatten()

