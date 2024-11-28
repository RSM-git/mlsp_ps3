from scipy.io import loadmat
from scipy.fft import idct, dct

import numpy as np
import matplotlib.pyplot as plt

data = loadmat("mlsp_ps3/problem3_1.mat")

idx = data["n"].flatten()
x = data["x"].flatten()

plt.scatter(idx, x)
plt.ylabel("$x(n)$")
plt.xlabel("$n$")
plt.show()

N = 2 ** 5 
l = 2 ** 9

assert N == len(x), f"{N=} should match the number of samples"


Phi = dct(np.eye(l), axis=0, norm='ortho')

B = np.zeros((N, l))

for i in range(N):
    B[i, idx[i]] = 1

print(B.shape, x.shape)
y = B @ x

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax[0].plot(x)
ax[0].plot(idx, y, 'r.')
ax[0].set_title('Original signal, time-domain')
ax[1].stem(idct(x, norm='ortho'))
ax[1].set_title('Original signal, IDCT-domain')
fig.tight_layout()
plt.show()