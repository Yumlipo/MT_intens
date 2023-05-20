import os,glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
import math

i = 0
power = []


def exp_fit(x, A, tau):
   return A * np.exp(-x / tau)

for filename in glob.glob("output/*/I-BG(t).txt", recursive=True):
   with open(filename, 'r') as f:
      data = np.loadtxt(filename)
      if i == 0:
         I = np.empty((0, data.shape[1]))
         i = 1
      dir = os.path.split(os.path.dirname(filename))[1]
      if "100" in dir:
         power += [100 for i in range(0, data.shape[0])]
      elif "50" in dir:
         power += [50 for i in range(0, data.shape[0])]
      elif "25" in dir:
         power += [25 for i in range(0, data.shape[0])]

      I = np.append(I, data, axis=0)
      for i in range(0, data.shape[0]):
         print (filename)

print(power)
print(I.shape)

tau = np.array([])
A = np.array([])
i = 0

# fig = plt.figure()
for cur_I in I:
   cur_I = cur_I[20:]
   t = np.linspace(0, cur_I.shape[0], cur_I.shape[0])
   param_optimised, param_covariance_matrix = curve_fit(exp_fit, t, cur_I, p0=[100, 100], maxfev=5000)
   param_t = param_optimised[1]
   param_A = param_optimised[0]

   tau = np.append(tau, param_t)
   A = np.append(A, param_A)


   # plt.plot(t, cur_I, label="I")
   # plt.plot(t, exp_fit(t, param_A, param_t), label='Exp fit')
   # plt.xlabel('frames, x ms')
   # plt.ylabel('I')
   # plt.legend()
   # plt.suptitle(f"Intensity from time for №{i + 1} MT with {power[i]}%")
   # fig.savefig("output/" + f"figure_{i+1}_{power[i]}%.png", bbox_inches="tight")
   # plt.clf()
   i += 1

print("tau", tau)
print("A", A)
print("size tau, A", tau.shape, A.shape)

with open("output/" + "params.txt", "wb") as f:
   np.savetxt(f, np.stack([tau, A], axis=1))
with open("output/" + "power.txt", "wb") as f:
   np.savetxt(f, power)
