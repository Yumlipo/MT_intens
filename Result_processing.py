#
#general processing of final results
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


arr = np.loadtxt('output\\params.txt').flatten()
tau = arr[::2]
A = arr[1::2]

# err = np.loadtxt('output\\err.txt').flatten()
# tau_err = err[::3]
# I_err = err[1::3]

# P = np.array([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
P = np.loadtxt('output\\power.txt')

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
axes[0].scatter(P, tau, label='tau')

axes[0].set_xlabel('Laser power, %')
axes[0].set_ylabel('y.e.')
axes[1].scatter(P, A, label='I0')

axes[1].set_xlabel('Laser power, %')
axes[1].set_ylabel('y.e.')
plt.suptitle("Params power dependence")

fit_tau = stats.linregress(P, tau)
axes[0].plot(P, fit_tau.intercept + fit_tau.slope*P, 'g', label='fitted tau')
axes[0].legend()

fit_I = stats.linregress(P, A)
axes[1].plot(P, fit_I.intercept + fit_I.slope*P, 'r', label='fitted I0')
axes[1].legend()

print("Tau fitting:\nslope = ", round(fit_tau.slope, 2), " intercept = ", round(fit_tau.intercept, 2))
print("I0 fitting:\nslope = ", round(fit_I.slope, 2), " intercept = ", round(fit_I.intercept, 2))

# plt.errorbar(P, tau, yerr=tau_err)
# plt.errorbar(P, I0, yerr=I_err)
fig.savefig("output/" + f"result.png", bbox_inches="tight")
plt.legend()
plt.show()