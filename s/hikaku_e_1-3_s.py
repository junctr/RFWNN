import csv
import numpy as np
from matplotlib import pyplot as plt

s = 4

t_data = np.loadtxt("time20.csv")
e1_data = np.loadtxt(f"s{s}_e1.csv")
e2_data = np.loadtxt(f"s{s}_e2.csv")
e3_data = np.loadtxt(f"s{s}_e3.csv")
s_data = np.loadtxt(f"s{s}_s.csv")
e1_z_data = np.loadtxt(f"z_s{s}_e1.csv")
e2_z_data = np.loadtxt(f"z_s{s}_e2.csv")
e3_z_data = np.loadtxt(f"z_s{s}_e3.csv")
s_z_data = np.loadtxt(f"z_s{s}_s.csv")

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.plot(t_data, e1_data, label = f"s{s}")
ax2.plot(t_data, e2_data, label = f"s{s}")
ax3.plot(t_data, e3_data, label = f"s{s}")
ax4.plot(t_data, s_data, label = f"s{s}")
ax1.plot(t_data, e1_z_data, label = f"s{s}_z")
ax2.plot(t_data, e2_z_data, label = f"s{s}_z")
ax3.plot(t_data, e3_z_data, label = f"s{s}_z")
ax4.plot(t_data, s_z_data, label = f"s{s}_z")

#ax1.xlim(0,20)
ax1.legend()
ax1.grid()
#ax2.xlim(0,20)
ax2.legend()
ax2.grid()
#ax3.xlim(0,20)
ax3.legend()
ax3.grid()
#ax4.xlim(0,20)
ax4.legend()
ax4.grid()


plt.title(f"seed{s}_z")

plt.show()