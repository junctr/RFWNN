import csv
import numpy as np
from matplotlib import pyplot as plt

s = 4
bz = 0.25

t_data = np.loadtxt("time100.csv")
#e_data = np.loadtxt(f"bz_s{s}_e_all.csv")
#e_all = np.load(f"bz_s{s}_e_all.npy",allow_pickle=True)

e_all_bz = np.loadtxt(f"bz_s{s}_e_all.csv")
e_all_nasi = np.loadtxt(f"nasi_s{s}_e_all.csv")

t_data = np.loadtxt("time100.csv")
e1_bz_data = e_all_bz[0]
e2_bz_data = e_all_bz[1]
e3_bz_data = e_all_bz[2]
e1_nasi_data = e_all_nasi[0]
e2_nasi_data = e_all_nasi[1]
e3_nasi_data = e_all_nasi[2]

fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(t_data, e1_nasi_data, label = "Conventional")
ax2.plot(t_data, e2_nasi_data, label = "Conventional")
ax3.plot(t_data, e3_nasi_data, label = "Conventional")

ax1.plot(t_data, e1_bz_data, label = f"Proposed_bz{bz}")
ax2.plot(t_data, e2_bz_data, label = f"Proposed_bz{bz}")
ax3.plot(t_data, e3_bz_data, label = f"Proposed_bz{bz}")



#ax1.xlim(0,20)
ax1.legend()
ax1.grid()
#ax2.xlim(0,20)
ax2.legend()
ax2.grid()
#ax3.xlim(0,20)
ax3.legend()
ax3.grid()

plt.show()