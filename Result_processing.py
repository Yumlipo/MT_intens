#
#general processing of final results
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


arr = np.loadtxt('output\\out.txt')
tau = arr[::2]
I0 = arr[1::2]

P = np.array([10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100])
plt.scatter(P, tau, label='tau')
plt.scatter(P, I0, label='I0')

fit_tau = stats.linregress(P, tau)
plt.plot(P, fit_tau.intercept + fit_tau.slope*P, 'g', label='fitted tau')

fit_I = stats.linregress(P, I0)
plt.plot(P, fit_I.intercept + fit_I.slope*P, 'r', label='fitted I0')

print("Tau fitting:\nslope = ", round(fit_tau.slope, 2), " intercept = ", round(fit_tau.intercept, 2))
print("I0 fitting:\nslope = ", round(fit_I.slope, 2), " intercept = ", round(fit_I.intercept, 2))

plt.legend()
plt.show()