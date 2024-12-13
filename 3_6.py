from scipy.io import loadmat
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.svm import SVR

import numpy as np

data = loadmat("problem3_6.mat")


t = data["t"].flatten()
y = data["y"].flatten()

print(t)
print(y)
center_idx = np.argmax(y)
print(f"Center location: t={t[center_idx]}")
# plt.plot(t, y, color="b", label="Original signal")
plt.xlabel("t")
plt.ylabel("y")
# plt.title("Chirp signal")
# plt.savefig("chirp_signal.pdf")
# plt.show()


def get_kernel_matrix(x: npt.NDArray, sigma2: float) -> npt.NDArray:
    N = len(x)
    pair_dist = x.reshape(-1, 1) - x.reshape(1, -1)
    K = np.exp(-1 / (2 * sigma2) * pair_dist**2)
    return K


def k_ridge(K: npt.NDArray, C: float, y: npt.NDArray):
    return np.linalg.solve(K + C * np.identity(K.shape[0]), y)


def k_ridge_pred(theta: npt.NDArray, K):
    return K @ theta


# sigma2 = 1
sigma2 = 0.5
C = 0.1
# C = 2
K = get_kernel_matrix(t, sigma2)
theta = k_ridge(K, C, y)
preds = k_ridge_pred(theta, K)
center_idx = np.argmax(preds)
print(f"Center location for ridge: t={t[center_idx]}")
# plt.plot(t, preds, color="r", label="Kernel ridge regression")
# plt.title("Original signal with the Kernel ridge regression signal")
# plt.legend()
# plt.savefig("3_6_krr.pdf")
# plt.show()
noise = y - preds
# plt.plot(t, noise)
# plt.show()

# snr_db = 10*np.log10(y.sum()/noise.sum())
# print(snr_db)

P_y = np.mean(y**2)
P_n = np.mean(noise**2)
print(P_y / P_n)

epsilon = 0.003
epsilon = 0.001
# epsilon = 0.1
kernel_params = 0.004
kernel_params = 0.9
gamma = 1 / (np.square(kernel_params))
C = 1
C = 0.8
regressor = SVR(kernel="rbf", gamma=gamma, C=C, epsilon=epsilon)
regressor.fit(t.reshape(-1, 1), y.reshape(-1, 1))
print(regressor.n_support_)
y_pred = regressor.predict(t.reshape(-1, 1))
outliers = np.abs(y_pred - y) > 1.5*epsilon
# t_nout = t[~outliers]
y_nout = y[~outliers]
noise_nouts = noise[~outliers]
P_y = np.mean(y_nout**2)
P_n = np.mean(noise_nouts**2)
print(P_y / P_n)

plt.plot(t, y, color="b", label="Original signal")
plt.plot(t, y_pred, color="r", label="Support vector regression")
plt.legend()
plt.title("Original signal with the Support vector regression signal")
plt.savefig("3_6_svr.pdf")
plt.show()


