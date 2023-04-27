import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import math


def gauss(x, C, x_mean, sigma):
    return C * exp(-(x - x_mean) ** 2 / (2 * sigma ** 2))

def exp_fit(x, A, tau, y0):
    return A * exp(-x/tau) + y0

#sliding window data smoothing
def smoothing(y, flag):
    # plt.plot(x, y, label="noise data")
    yhat = signal.savgol_filter(y, 50, 3)

    # if flag == 1:
        # myhist(yhat)
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

    # plt.scatter(bins[:-1], n, color="red")
    #
    #
    #
    # mean = sum(bins[:-1] * n) / sum(n)
    # sigma = sum(n * (bins[:-1] - mean) ** 2) / sum(n)
    # param_optimised, param_covariance_matrix = curve_fit(gauss, bins[:-21], n[-20], p0=[max(n), mean, sigma], maxfev=5000)
    # x_hist_2 = np.linspace(np.min(bins[:-1]), np.max(bins[:-1]), 500)
    #
    # print("param", param_optimised)
    # plt.plot(bins, gauss(bins, param_optimised[0], param_optimised[1], param_optimised[2]), label='Gaussian fit')

#Draw I(t) and get tau parametr from fitting this dependence
def draw_results_and_param(IminusBG_arr, I_point_arr, I_BG_arr):
    # num_of_points = IminusBG_arr.shape[1]
    tau = np.array([])
    I0 = np.array([])
    y0 = np.array([])
    param_cov = np.array([])
    t = np.linspace(0, IminusBG_arr.shape[1], IminusBG_arr.shape[1])

    # for i in range(IminusBG_arr.shape[0]):
    #     plt.plot(t, IminusBG_arr[i], label=str(i))
    # plt.xlabel('frames, x ms')
    # plt.ylabel('I(point) - I(BG)')
    # plt.title("Signal to noise from time")
    # plt.legend()
    # plt.show()
    for i in range(IminusBG_arr.shape[0]):
        IminusBG_arr[i] = smoothing(IminusBG_arr[i], 1)
        I_point_arr[i] = smoothing(I_point_arr[i], 0)
        I_BG_arr[i] = smoothing(I_BG_arr[i], 0)

        param_t, param_I, param_y0, param_covariance_matrix = get_tau(IminusBG_arr[i].flatten(), t)
        tau = np.append(tau, param_t)
        I0 = np.append(I0, param_I)
        y0 = np.append(y0, param_y0)
        param_cov =  np.append(param_cov, param_covariance_matrix)

        fig, ax = plt.subplots()
        ax.plot(t, IminusBG_arr[i], label="I(MT) - I(BG) " + str(i + 1))
        ax.plot(t, I_point_arr[i], label="MT " + str(i + 1))
        ax.plot(t, I_BG_arr[i], label="BG " + str(i + 1))
        ax.plot(t, exp_fit(t, param_I, param_t, param_y0), label='Exp fit')
        ax.set_xlabel('frames, x ms')
        ax.set_ylabel('I')
        ax.set_title("Signal to noise from time for №" + str(i + 1) + " MT")
        ax.legend()

    return tau, I0, y0, param_cov

def get_tau(IminusBG, t):
    param_optimised, param_covariance_matrix = curve_fit(exp_fit, t, IminusBG, p0=[10, 100, 20], maxfev = 5000)
    return round(param_optimised[1]), round(param_optimised[0]), round(param_optimised[2]), np.sqrt(np.diag(param_covariance_matrix))

#I don't rotate each img from stack, I’m just taking every pixel between the lines that define the rectangle
def int_from_rect(x1, y1, x2, y2, img_process):
    l = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    if x2 < x1:
        tmp = x1
        x1 = x2
        x2 = tmp

        tmp = y1
        y1 = y2
        y2 = tmp

    a = (y2 - y1) / (x2 - x1) #slope angle

    if a > -0.01 and a < 0.01:#if slope angle is nearly 0
        x3 = x1
        y3 = y1 - 3
        x4 = x2
        y4 = y2 + 3
        copy_img = img_process[int(y3):int(y4), int(x3):int(x4)].copy()
        return np.sum(copy_img) / l

    else:# else we need the coordinates of all the corners of the rectangle (see the notation at the beginning of the window.py file)
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
            if xx >= line1(yy, a, x1, y1) and xx >= line2(yy, a, x4, y4) and xx <= line2(yy, a, x3, y3) and xx <= line1(yy, a, x2, y2):

                Intens += img_process[yy, xx]
                # img[yy, xx] = (255, 255, 0)

    return Intens / l


def line1(yy, a, x_param, y_param):
    # x=1/a*y-1/a*b
    return -a * yy + x_param + a * y_param


def line2(yy, a, x_param, y_param):
    return yy / a - y_param / a + x_param
