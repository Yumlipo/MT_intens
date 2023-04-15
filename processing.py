import numpy as np
from math import factorial
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline
from scipy import stats

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import math

def smoothing(y, flag):
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



def int_from_rect(crds, img_process):
    x1, y1, x2, y2 = crds
    l = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    if x2 < x1:
        tmp = x1
        x1 = x2
        x2 = tmp

        tmp = y1
        y1 = y2
        y2 = tmp

    a = (y2 - y1) / (x2 - x1)

    if a > -0.01 and a < 0.01:
        x3 = x1
        y3 = y1 - 3
        x4 = x2
        y4 = y2 + 3
        copy_img = img_process[int(y3):int(y4), int(x3):int(x4)].copy()
        return np.sum(copy_img) / l

    else:
        x3 = int(round(x1 + 3 / math.sqrt(1 + 1 / (a * a))))
        y3 = int(round(-x3 / a + x1 / a + y1))

        x4 = int(round(x2 - 3 / math.sqrt(1 + 1 / (a * a))))
        y4 = int(round(-x4 / a + x2 / a + y2))

        x5 = int(round(x1 - 3 / math.sqrt(1 + 1 / (a * a))))
        y5 = int(round(-x5 / a + x1 / a + y1))

        x6 = int(round(x2 + 3 / math.sqrt(1 + 1 / (a * a))))
        y6 = int(round(-x6 / a + x2 / a + y2))

    y_array = np.array([y3, y4, y5, y6])
    x_array = np.array([x3, x4, x5, x6])

    y_min = np.min(y_array)
    y_max = np.max(y_array)

    x_min = np.min(x_array)
    x_max = np.max(x_array)

    Intens = 0
    for yy in np.arange(y_min, y_max):
        for xx in np.arange(x_min, x_max):
            if xx > line1(yy, a, x1, y1) and xx > line3(yy, a, x4, y4) and xx < line2(yy, a, x3, y3) and xx < line4(yy, a, x2, y2):

                Intens += img_process[yy, xx]
                # img[yy, xx] = (255, 255, 0)

    return Intens / l


def line1(yy, a, x1, y1):
    # x=1/a*y-1/a*b
    return -a * yy + x1 + a * y1


def line2(yy, a, x3, y3):
    return yy / a - y3 / a + x3


def line3(yy, a, x4, y4):
    return yy / a - y4 / a + x4


def line4(yy, a, x2, y2):
    return -a * yy + x2 + a * y2