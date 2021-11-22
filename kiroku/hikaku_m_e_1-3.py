import csv
import numpy as np
from matplotlib import pyplot as plt

n_seed = 4

m1 = 0.0
m2 = 0.01
m3 = 0.1
m4 = 0.2
m5 = 0.3
m6 = 0.4

T = 1000
end = 25

t_data = np.loadtxt(f"time{end}.csv")
#e_data = np.loadtxt(f"p_s{s}_e_all.csv")
#e_all = np.load(f"p_s{s}_e_all.npy",allow_pickle=True)

e_all_m1 = np.loadtxt(f"k_s{n_seed}_m{m1}_T{T}_t{end}_e_all.csv")
e_all_m2 = np.loadtxt(f"k_s{n_seed}_m{m2}_T{T}_t{end}_e_all.csv")
e_all_m3 = np.loadtxt(f"k_s{n_seed}_m{m3}_T{T}_t{end}_e_all.csv")
e_all_m4 = np.loadtxt(f"k_s{n_seed}_m{m4}_T{T}_t{end}_e_all.csv")
e_all_m5 = np.loadtxt(f"k_s{n_seed}_m{m5}_T{T}_t{end}_e_all.csv")
e_all_m6 = np.loadtxt(f"k_s{n_seed}_m{m6}_T{T}_t{end}_e_all.csv")


e1_m1_data = e_all_m1[0]
e2_m1_data = e_all_m1[1]
e3_m1_data = e_all_m1[2]
e1_m2_data = e_all_m2[0]
e2_m2_data = e_all_m2[1]
e3_m2_data = e_all_m2[2]
e1_m3_data = e_all_m3[0]
e2_m3_data = e_all_m3[1]
e3_m3_data = e_all_m3[2]
e1_m4_data = e_all_m4[0]
e2_m4_data = e_all_m4[1]
e3_m4_data = e_all_m4[2]
e1_m5_data = e_all_m5[0]
e2_m5_data = e_all_m5[1]
e3_m5_data = e_all_m5[2]
e1_m6_data = e_all_m6[0]
e2_m6_data = e_all_m6[1]
e3_m6_data = e_all_m6[2]

fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)


ax1.plot(t_data, e1_m1_data, label = f"s{n_seed}_m{m1}_T{T}")
ax2.plot(t_data, e2_m1_data, label = f"s{n_seed}_m{m1}_T{T}")
ax3.plot(t_data, e3_m1_data, label = f"s{n_seed}_m{m1}_T{T}")

ax1.plot(t_data, e1_m2_data, label = f"s{n_seed}_m{m2}_T{T}")
ax2.plot(t_data, e2_m2_data, label = f"s{n_seed}_m{m2}_T{T}")
ax3.plot(t_data, e3_m2_data, label = f"s{n_seed}_m{m2}_T{T}")

ax1.plot(t_data, e1_m3_data, label = f"s{n_seed}_m{m3}_T{T}")
ax2.plot(t_data, e2_m3_data, label = f"s{n_seed}_m{m3}_T{T}")
ax3.plot(t_data, e3_m3_data, label = f"s{n_seed}_m{m3}_T{T}")

ax1.plot(t_data, e1_m4_data, label = f"s{n_seed}_m{m4}_T{T}")
ax2.plot(t_data, e2_m4_data, label = f"s{n_seed}_m{m4}_T{T}")
ax3.plot(t_data, e3_m4_data, label = f"s{n_seed}_m{m4}_T{T}")

ax1.plot(t_data, e1_m5_data, label = f"s{n_seed}_m{m5}_T{T}")
ax2.plot(t_data, e2_m5_data, label = f"s{n_seed}_m{m5}_T{T}")
ax3.plot(t_data, e3_m5_data, label = f"s{n_seed}_m{m5}_T{T}")

ax1.plot(t_data, e1_m6_data, label = f"s{n_seed}_m{m6}_T{T}")
ax2.plot(t_data, e2_m6_data, label = f"s{n_seed}_m{m6}_T{T}")
ax3.plot(t_data, e3_m6_data, label = f"s{n_seed}_m{m6}_T{T}")


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