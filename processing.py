import numpy as np
from math import factorial
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline
from scipy import stats

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def smoothing(y, x, flag):
    # plt.plot(x, y, label="noise data")
    yhat = signal.savgol_filter(y, 50, 3)

    if flag == 1:
        myhist(yhat)
    # plt.plot(x, y)
    # plt.plot(x, yhat, color='green', label="processing data")
    # plt.legend()
    # plt.show()
    return yhat


def myhist(arr):
    n, bins, bars_container = plt.hist(arr, bins=50, density=True, rwidth=0.9)
    plt.title("Intensity histogram")
    plt.xlabel("I")
    plt.ylabel("bins")

    # print("arr", arr)
    # print("n", n)
    # print("bins", bins)
    # print("b_c", bars_container)

    plt.scatter(bins[:-1], n, color="red")

    def gauss(x, C, x_mean, sigma):
        return C * exp(-(x - x_mean) ** 2 / (2 * sigma ** 2))

    mean = sum(bins[:-1] * n) / sum(n)
    sigma = sum(n * (bins[:-1] - mean) ** 2) / sum(n)
    param_optimised, param_covariance_matrix = curve_fit(gauss, bins[:-21], n[-20], p0=[max(n), mean, sigma], maxfev=5000)
    x_hist_2 = np.linspace(np.min(bins[:-1]), np.max(bins[:-1]), 500)

    print("param", param_optimised)
    plt.plot(bins, gauss(bins, param_optimised[0], param_optimised[1], param_optimised[2]), label='Gaussian fit')