import csv
import numpy as np
from matplotlib import pyplot as plt

s = 22
m = 0.5
e = 2

t_data = np.loadtxt("time20.csv")
m_data = np.loadtxt(f"s{s}_m{m}_e{e}.csv")
nasi_data = np.loadtxt(f"s{s}_m0.0_e{e}.csv")

plt.plot(t_data, nasi_data, color="tab:green", label = "Conventional")
plt.plot(t_data, m_data, color="tab:red", label = "Proposed")
plt.xlabel("time (s)")
plt.ylabel(f"tracking error of link {e} (rad)")
plt.xlim(0,20)
plt.legend()
plt.grid()
plt.show()