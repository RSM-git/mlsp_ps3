import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Callable

def ICA(x, mu, num_components, iters, mode):
    # Random initialization
    W = np.random.rand(num_components, num_components)
    N = np.size(x, 1)

    if mode == 'super':
        phi = lambda u : 2*np.tanh(u)
    elif mode == 'sub':
        phi = lambda u : u-np.tanh(u)
    else:
        print("Unknown mode")
        return W


    for i in range(iters):
        z = W @ x # num_components x N

        u = phi(z)@z.T / N # The expectation E[phi(z)z^T]
        dW = (np.eye(num_components) - u) @ W #  
        # Update
        W = W + mu*dW   
    
    # We return W and to get Â we will invert W
    return W

def estimate_error(A, A_hat):
    # Max normalization of the matrices
    A = A / np.linalg.norm(A, axis=0)
    A_hat = A_hat / np.linalg.norm(A_hat, axis=0)

    # permutations = [np.linalg.norm(A - A_hat), np.linalg.norm(A - A_hat[::-1]),
    #                 np.linalg.norm(A - A_hat[:, ::-1]), np.linalg.norm(A - A_hat[::-1, ::-1])] Includes row permutations and row+column permutations
    permutations = [np.linalg.norm(A - A_hat), np.linalg.norm(A - A_hat[:, ::-1])] # Only permutes the columns.

    error = np.min(permutations)


    return np.min(permutations), [A_hat, A_hat[:, ::-1]][np.argmin(permutations)]


def run_experiment(experiment_num: int = 0, iterations: int = 100, N: int = 2500, ICA_param_dict = {"mu": 0.1, "iterations": 200, "components": 2, "gaussianity": "sub"}):
    A = np.array([[3, 1],[1, 1]])

    mu = ICA_param_dict["mu"]
    iters = ICA_param_dict["iterations"]
    components = ICA_param_dict["components"]
    gaussianity = ICA_param_dict["gaussianity"]

    errors = []

    for _ in range(iterations):
        match experiment_num:
            case 0:
                s = np.random.rand(2, N)
            case 1:
                # s1 drawn from U(0, 1); s2 drawn from Beta(0.1, 0.1)
                s1 = np.random.rand(N)
                s2 = np.random.beta(0.1, 0.1, N)
                s = np.stack([s1, s2],axis=0)
            case 2:
                # s1 drawn frm U(0, 1); s2 drawn from N(0,1)
                s1 = np.random.rand(N)
                s2 = np.random.randn(N)
                s = np.stack([s1, s2],axis=0)
            case 3:
                # s drawn from a multivariate normal distribution with mean [0, 1] and covariance [[2, 0.25],[0.25, 1]]
                s = np.random.multivariate_normal([0, 1], [[2, 0.25],[0.25, 1]], size=N).T # Transpose to get 2 x N
            case _:
                Exception("Unsupported experiment.")


        x = A @ s

        # Standardize the observations
        col_means = np.mean(x, axis=1)
        x = x - col_means[:,None]

        W = ICA(x, mu, components, iters, gaussianity)

        # W = W / np.max(W)

        error, A_hat = estimate_error(A, np.linalg.inv(W))
        errors.append(error)


    plt.figure(figsize=(12,4))
    plt.plot(errors)
    plt.title(f"Error across {iterations} repetitions.")
    # plt.savefig(f"experiment_{experiment_num}_error.pdf")
    plt.show()

    # Compute estimated sources (unmixed signals)
    y_hat = np.linalg.inv(A_hat) @ x

    plt.figure(figsize=(16,4))

    plt.subplot(1, 3, 1)
    plt.scatter(s[0, :], s[1, :], s=2)
    plt.title("Sources $(s)$")

    plt.subplot(1, 3, 2)
    plt.scatter(x[0, :], x[1, :], s=2)
    plt.title("Observations $(As)$")


    plt.subplot(1, 3, 3)
    plt.scatter(y_hat[0,:],y_hat[1,:], s=2)
    plt.title("Estimated sources $(\hat{A}^{-1}x)$")
    # plt.savefig(f"experiment_{experiment_num}_signals.pdf")
    plt.show()


if __name__ == '__main__':
    run_experiment(experiment_num=0, N=5000)
    run_experiment(experiment_num=1, N=5000)
    run_experiment(experiment_num=2, N=5000)
    run_experiment(experiment_num=3, N=5000, ICA_param_dict= {"mu": 0.01, "iterations": 200, "components": 2, "gaussianity": "sub"})