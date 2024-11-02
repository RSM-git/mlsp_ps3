from scipy.io import loadmat
from scipy.fft import idct, dct

import numpy as np
import matplotlib.pyplot as plt

data = loadmat("problem3_1.mat")

idx = data["n"].flatten()
vals = data["x"].flatten()


plt.scatter(idx, vals)
plt.ylabel("$x(n)$")
plt.xlabel("$n$")
plt.show()
