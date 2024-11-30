import numpy as np
import matplotlib.pyplot as plt

# Set the parameters
q = 1
dt = 0.1
s = 0.5
F = np.array([
    [1, dt],
    [0, 1],
])

Q = q*np.array([
    [dt**3/3, dt**2/2],
    [dt**2/2, dt],
])

H = np.array([
    [1, 0],
])
R = s**2

x0 = np.array([[0], [1]])

np.random.seed(2)

raw_errors = []
kf_errors = []
n_experiments = 1_000
for experiment in range(n_experiments):
    steps = 10
    X = np.zeros((len(F), steps))
    Y = np.zeros((len(H), steps))
    x = x0
    for k in range(steps):
        q = np.linalg.cholesky(Q)@np.random.randn(len(F), 1)
        x = F@x + q
        y = H@x + s*np.random.randn(1, 1)
        X[:, k] = x[:, 0]
        Y[:, k] = y[:, 0]



    # Kalman filter
    P0 = np.identity(2)
    x = x0
    P = P0
    kf_x = np.zeros((len(x), Y.shape[1]))
    kf_P = np.zeros((len(P), P.shape[1], Y.shape[1]))
    for k in range(Y.shape[1]):
        x = F@x
        P = F@P@F.T + Q

        e = Y[:, k].reshape(-1, 1) - H@x
        S = H@P@H.T + R
        K = P@H.T@np.linalg.inv(S)
        x = x + K@e
        P = P - K@S@K.T

        kf_x[:, k] = x[:, 0]
        kf_P[:, :, k] = P

    rmse_raw = np.sqrt(np.mean(np.sum((Y - X[0, :])**2, 1)))
    rmse_kf = np.sqrt(np.mean((kf_x[0, :] - X[0, :])**2))
    raw_errors.append(rmse_raw)
    kf_errors.append(rmse_kf)

print("Raw errors stats")
print("Mean: ", np.round(np.mean(raw_errors),2))
print("Std: ", np.round(np.std(raw_errors),2))
interval_raw = np.percentile(raw_errors, [2.5, 97.5])
print(f"Confidence interval: ({interval_raw[0]:.2f}, {interval_raw[1]:.2f})")
print()

print("KF errors stats")
print("Mean: ", np.round(np.mean(kf_errors),2))
print("Std: ", np.round(np.std(kf_errors),2))
interval_kf = np.percentile(kf_errors, [2.5, 97.5])
print(f"Confidence interval: ({interval_kf[0]:.2f}, {interval_kf[1]:.2f})")

plt.figure()
plt.plot(range(steps), X[0, :], '-')
plt.plot(range(steps), Y[0, :], '.')
# plt.plot(X[0, 0], X[1, 0], 'x')
plt.legend(['Trajectory', 'Measurements'])
plt.xlabel('Time step')
plt.ylabel('Position')
plt.savefig("3_5_data.pdf")
plt.show()

plt.figure()
plt.plot(range(steps), X[0, :], '-')
plt.plot(range(steps), Y[0, :], 'o')
plt.plot(range(steps), kf_x[0, :], '-')
plt.legend(['True Trajectory', 'Measurements', 'Filter Estimate'])
plt.xlabel('Time step')
plt.ylabel('Position')
plt.savefig("3_5_kf.pdf")
plt.show()


# RTS smoother

# ms = kf_x[:, -1]
# Ps = kf_P[:, :, -1]
# rts_m = np.zeros((len(x), Y.shape[1]))
# rts_P = np.zeros((len(P), P.shape[1], Y.shape[1]))
# rts_m[:, -1] = ms
# rts_P[:, :, -1] = Ps
# for k in reversed(range(kf_x.shape[1])):
#     mp = F@kf_x[:, k]
#     Pp = F@kf_P[:, :, k]@F.T+Q
#     Gk = kf_P[:, :, k]@F.T@np.linalg.inv(Pp)
#     ms = kf_x[:, k] + Gk@(ms - mp)
#     Ps = kf_P[:, :, k] + Gk@(Ps - Pp)@Gk.T
#     rts_m[:, k] = ms
#     rts_P[:, :, k] = Ps

# rmse_rts = np.sqrt(np.mean(np.sum((rts_m[:2, :] - X[:2, :])**2, 1)))

# plt.figure()
# plt.plot(X[0, :], X[1, :], '-')
# plt.plot(Y[0, :], Y[1, :], 'o')
# plt.plot(rts_m[0, :], rts_m[1, :], '-')
# plt.legend(['True Trajectory', 'Measurements', 'Smoother Estimate'])
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')


# plt.show()
