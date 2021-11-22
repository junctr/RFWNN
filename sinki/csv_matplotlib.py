import csv
import numpy as np
from matplotlib import pyplot as plt

t_data = np.loadtxt("time20.csv")
m_data = np.loadtxt("m_e2.csv")
nasi_data = np.loadtxt("nasi_e2.csv")

plt.plot(t_data, nasi_data, label = "Conventional")
plt.plot(t_data, m_data, label = "Proposed")
plt.xlabel("time (s)")
plt.ylabel("tracking error for link 2 (rad)")
plt.xlim(0,20)
plt.legend()
plt.grid()
plt.show()