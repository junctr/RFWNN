import csv
import numpy as np
from matplotlib import pyplot as plt

s = 8

t_data = np.loadtxt("time100.csv")
#e_data = np.loadtxt(f"bz_s{s}_e_all.csv")
#e_all = np.load(f"bz_s{s}_e_all.npy",allow_pickle=True)

e_all = np.loadtxt(f"bz_s{s}_e_all.csv")
for i_e in range(21):
    plt.plot(t_data, e_all[i_e],label = f"{i_e}")


plt.xlabel("time (s)")
#plt.ylabel(f"tracking error of link {e} (rad)")
#plt.xlim(0,20)
plt.legend()
plt.grid()
plt.show()
