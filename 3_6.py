from scipy.io import loadmat
import numpy.typing as npt
import matplotlib.pyplot as plt

import numpy as np

data = loadmat("problem3_6.mat")


t = data["t"].flatten()
y = data["y"].flatten()

# print(t)
# print(y)
# center_idx = np.argmax(y)
# print(f"Center location: t={t[center_idx]}")
plt.plot(t, y, color="b")
# plt.show()

def get_kernel_matrix(x: npt.NDArray, sigma2: float) -> npt.NDArray:
    N = len(x)
    pair_dist = x.reshape(-1, 1) - x.reshape(1, -1)
    K = np.exp(-1/(2*sigma2)*pair_dist**2)
    return K


def k_ridge(K: npt.NDArray, C: float, y: npt.NDArray):
    return np.linalg.solve(K+C*np.identity(K.shape[0]), y)

def k_ridge_pred(theta: npt.NDArray, K):
    return K@theta

sigma2 = 1
# sigma2 = 0.0016
C = 0.001
K = get_kernel_matrix(t, sigma2)
theta = k_ridge(K, C, y)
preds = k_ridge_pred(theta, K)
plt.plot(t, preds, color="r")
plt.show()