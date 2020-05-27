import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from src.MNGM2 import MNGM2



def plot_confidence_ellipse_cov():

    np.random.seed(0)
    n = 1000
    x_0 = [0.1, 0.1]
    ungm = MNGM2(n, x_0)
    ungm.generate_data()

    sig = np.zeros((n, 4))  ## x0 x1 y0 y1
    sig_mean = np.zeros(4)  # mean(x0) mean(x1) mean(y0) mean(y1)
    for i in range(2):
        sig[:, i] = ungm.x[:, i]
        sig[:, i+2] = ungm.y[:, i]

        sig_mean[i] = np.mean(sig[:, i])
        sig_mean[i+2] = np.mean(sig[:, i+2])
    #y_mean = np.mean(x[:, 1])

    angle = np.zeros(2)
    lambda_ = np.zeros(4)
    # first ellipse:
    cov1 = np.cov(sig[:, 0], sig[:, 1])
    lambda_1, v = np.linalg.eig(cov1)
    lambda_1 = np.sqrt(lambda_1)
    lambda_[0:2] = lambda_1
    angle[0] = np.arctan2(*v[:, 0][::-1])

    # second ellipse:
    cov2 = np.cov(sig[:, 2], sig[:, 3])
    lambda_2, v = np.linalg.eig(cov2)
    lambda_2 = np.sqrt(lambda_2)
    lambda_[2:4] = lambda_2
    angle[1] = np.arctan2(*v[:, 0][::-1])

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    for i, ax in enumerate(axs):# ax in axs:
        ax.scatter(sig[:, i*2], sig[:, i*2+1], s=0.9)

        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)

        ell = Ellipse(xy=(sig_mean[i*2], sig_mean[i*2+1]),
                    width=lambda_[i*2] * 6, height=lambda_[i*2+1] * 6,
                    angle=np.rad2deg(angle[i]), edgecolor='firebrick', label=r'$3\sigma$')
        ell.set(label=r'$3\sigma$')
        ax.add_artist(ell)
        ell.set_facecolor('none')
        ax.set(xlim=[-10, 10], ylim=[-10, 10], aspect='equal')
        ax.scatter(sig_mean[i*2], sig_mean[i*2+1], c='red', s=3)
        if i == 0:
            ax.set_title('cov state var')
        else:
            ax.set_title('cov output var')
    plt.show()






plot_confidence_ellipse_cov()
