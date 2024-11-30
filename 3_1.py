from scipy.io import loadmat
from scipy.fft import idct, dct
from sklearn.linear_model import Lasso, LassoLars
import numpy as np
import matplotlib.pyplot as plt

algorithm = "OMP"

# Load data and setup DCT and sensing matrix
data = loadmat("problem3_1.mat")
idx = data["n"].flatten() - 1
x = data["x"].flatten()

N = 2 ** 5 
l = 2 ** 9

assert N == len(x), f"{N=} should match the number of samples"

Phi = dct(np.eye(l), axis=0, norm='ortho')
B = np.zeros((N, l))

for i in range(N):
    B[i, idx[i]] = 1

BF = B @ Phi

# Lasso / LassoLars
lambda_ = 0.005
model = Lasso(lambda_, fit_intercept=False)
model.fit(BF, x)
solsB = model.coef_


if algorithm == "IST":
    nsteps = 100000
    t_ = np.zeros((l, nsteps))
    mu = 0.1
    for k in range(1, nsteps):
        e = x - BF @ t_[:, k-1]
        t_tilde = t_[:, k-1] + mu * BF.T @ e
        t_[:, k] = np.sign(t_tilde) * np.maximum(np.abs(t_tilde)-lambda_*mu,0) # complete the line 
    solsIST = t_[:, -1]

    # plot solutions
    fig, ax= plt.subplots(1, 1, figsize=(6, 3))
    ax.stem(solsB, markerfmt='bo', label='sklearn Lasso', basefmt=' ')
    ax.stem(solsIST, markerfmt='ro', label='IST', basefmt=' ')
    ax.legend()
    ax.set_title('Solutions')
    plt.show()


    x_hat = Phi @ solsIST

    fig, ax= plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(x_hat)
    ax[0].plot(idx, x, ".", color="orange")
    ax[0].set_title('Estimated using randomly picked samples')
    

    # Rescale the coefficients.
    a_hat = np.sqrt(2 / l) * solsIST[solsIST > 1e-10]
    m_hat = np.where(solsIST > 1e-10)[0]+1

    print(f"K = {len(a_hat)}")
    print(f"a_hat = {np.round(a_hat,2)}")
    print(f"m_hat = {m_hat}")

    x_true = np.zeros(l)
    for i in range(4):
        x_true += a_hat[i]*np.cos(np.pi/(2*l)*(2*m_hat[i]-1)*np.arange(l))

    ax[1].plot(x_true)
    ax[1].plot(idx, x, ".", color="orange")
    ax[1].set_title("Signal reconstruction with coefficients and frequencies.")
    fig.tight_layout()
    
    plt.show()

elif algorithm == "OMP":
    # create OMP solution
    k_OMP = 4  # number of vectors
    X = BF  # set X
    # initialize
    residual = x
    S = np.zeros(k_OMP, dtype=int)
    normx = np.sqrt(np.sum(X**2, axis=0)) # shortcut formula
    for i in range(k_OMP):
        proj = X.T@residual/normx  # solution  
        pos = np.argmax(abs(proj))
        S[i] = pos
        Xi = X[:, S[:i+1]]
        theta_ = np.linalg.solve(Xi.T @ Xi, Xi.T @ x) # solution  
        theta = np.zeros(X.shape[1])
        theta[S[:i+1]] = theta_
        residual = x - X@theta # solution  
    
    solsOMP = theta


    # plot solutions
    fig, ax= plt.subplots(1, 1, figsize=(6, 3))
    ax.stem(solsB, markerfmt='bo', label='sklearn Lasso', basefmt=' ')
    ax.stem(solsOMP, markerfmt='ro', label='OMP', basefmt=' ')
    ax.legend()
    ax.set_title('Solutions')

    # Extract the indices of the K largest coefficients
    arg_idx = ind = np.argpartition(solsOMP, -k_OMP)[-k_OMP:]

    a_hat = solsOMP[arg_idx]*np.sqrt(2/l)
    m_hat = arg_idx + 1

    print(f"K = {k_OMP}")
    print(f"a_hat = {np.round(a_hat,2)}")
    print(f"m_hat = {m_hat}")

    # solsOMP is the estimated X, reconstruct x using the synthesis
    x_hat = Phi @ solsOMP

    fig, ax= plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(x_hat)
    ax[0].plot(idx, x, ".", color="orange")
    ax[0].set_title('Estimated using randomly picked samples')

    x_true = np.zeros(l)
    for i in range(4):
        x_true += a_hat[i]*np.cos(np.pi/(2*l)*(2*m_hat[i]-1)*np.arange(l))

    ax[1].plot(x_true)
    ax[1].plot(idx, x, ".", color="orange")
    ax[1].set_title("Signal reconstruction with coefficients and frequencies.")
    fig.tight_layout()
    plt.show()
