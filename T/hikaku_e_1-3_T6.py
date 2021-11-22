import csv
import numpy as np
from matplotlib import pyplot as plt

s = 4

m = 0.0

T1 = 1
T2 = 10
T3 = 100
T4 = 1000
T5 = 10000
T6 = 5000

t_data = np.loadtxt("time20.csv")
e1_T1_data = np.loadtxt(f"s{s}_m{m}_T{T1}_e1.csv")
e2_T1_data = np.loadtxt(f"s{s}_m{m}_T{T1}_e2.csv")
e3_T1_data = np.loadtxt(f"s{s}_m{m}_T{T1}_e3.csv")
e1_T2_data = np.loadtxt(f"s{s}_m{m}_T{T2}_e1.csv")
e2_T2_data = np.loadtxt(f"s{s}_m{m}_T{T2}_e2.csv")
e3_T2_data = np.loadtxt(f"s{s}_m{m}_T{T2}_e3.csv")
e1_T3_data = np.loadtxt(f"s{s}_m{m}_T{T3}_e1.csv")
e2_T3_data = np.loadtxt(f"s{s}_m{m}_T{T3}_e2.csv")
e3_T3_data = np.loadtxt(f"s{s}_m{m}_T{T3}_e3.csv")
e1_T4_data = np.loadtxt(f"s{s}_m{m}_T{T4}_e1.csv")
e2_T4_data = np.loadtxt(f"s{s}_m{m}_T{T4}_e2.csv")
e3_T4_data = np.loadtxt(f"s{s}_m{m}_T{T4}_e3.csv")
e1_T5_data = np.loadtxt(f"s{s}_m{m}_T{T5}_e1.csv")
e2_T5_data = np.loadtxt(f"s{s}_m{m}_T{T5}_e2.csv")
e3_T5_data = np.loadtxt(f"s{s}_m{m}_T{T5}_e3.csv")
e1_T6_data = np.loadtxt(f"s{s}_m{m}_T{T6}_e1.csv")
e2_T6_data = np.loadtxt(f"s{s}_m{m}_T{T6}_e2.csv")
e3_T6_data = np.loadtxt(f"s{s}_m{m}_T{T6}_e3.csv")

fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(t_data, e1_T1_data, label = f"T{T1}")
ax2.plot(t_data, e2_T1_data, label = f"T{T1}")
ax3.plot(t_data, e3_T1_data, label = f"T{T1}")
ax1.plot(t_data, e1_T2_data, label = f"T{T2}")
ax2.plot(t_data, e2_T2_data, label = f"T{T2}")
ax3.plot(t_data, e3_T2_data, label = f"T{T2}")
ax1.plot(t_data, e1_T3_data, label = f"T{T3}")
ax2.plot(t_data, e2_T3_data, label = f"T{T3}")
ax3.plot(t_data, e3_T3_data, label = f"T{T3}")
ax1.plot(t_data, e1_T4_data, label = f"T{T4}")
ax2.plot(t_data, e2_T4_data, label = f"T{T4}")
ax3.plot(t_data, e3_T4_data, label = f"T{T4}")
ax1.plot(t_data, e1_T5_data, label = f"T{T5}")
ax2.plot(t_data, e2_T5_data, label = f"T{T5}")
ax3.plot(t_data, e3_T5_data, label = f"T{T5}")
ax1.plot(t_data, e1_T6_data, label = f"T{T6}")
ax2.plot(t_data, e2_T6_data, label = f"T{T6}")
ax3.plot(t_data, e3_T6_data, label = f"T{T6}")

#ax1.xlim(0,20)
ax1.legend()
ax1.grid()
#ax2.xlim(0,20)
ax2.legend()
ax2.grid()
#ax3.xlim(0,20)
ax3.legend()
ax3.grid()

plt.title(f"seed{s}_m{m}")

plt.show()