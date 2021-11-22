import csv
import numpy as np
from matplotlib import pyplot as plt

n_seed = 4

m = 0.0

az1 = 0.025
az2 = 0.05
az3 = 0.1
az4 = 0.2
az5 = 0.4
#az6 = 0.0

T = 1000
end = 25

t_data = np.loadtxt(f"time{end}.csv")
#e_data = np.loadtxt(f"p_s{s}_e_all.csv")
#e_all = np.load(f"p_s{s}_e_all.npy",allow_pickle=True)

e_all_az1 = np.loadtxt(f"k_s{n_seed}_m{m}_T{T}_t{end}_az{az1}_e_all.csv")
e_all_az2 = np.loadtxt(f"k_s{n_seed}_m{m}_T{T}_t{end}_az{az2}_e_all.csv")
e_all_az3 = np.loadtxt(f"k_s{n_seed}_m{m}_T{T}_t{end}_az{az3}_e_all.csv")
e_all_az4 = np.loadtxt(f"k_s{n_seed}_m{m}_T{T}_t{end}_az{az4}_e_all.csv")
e_all_az5 = np.loadtxt(f"k_s{n_seed}_m{m}_T{T}_t{end}_az{az5}_e_all.csv")
#e_all_az6 = np.loadtxt(f"k_s{n_seed}_m{m}_T{T}_t{end}_az{az6}_e_all.csv")


e1_az1_data = e_all_az1[0]
e2_az1_data = e_all_az1[1]
e3_az1_data = e_all_az1[2]
e1_az2_data = e_all_az2[0]
e2_az2_data = e_all_az2[1]
e3_az2_data = e_all_az2[2]
e1_az3_data = e_all_az3[0]
e2_az3_data = e_all_az3[1]
e3_az3_data = e_all_az3[2]
e1_az4_data = e_all_az4[0]
e2_az4_data = e_all_az4[1]
e3_az4_data = e_all_az4[2]
e1_az5_data = e_all_az5[0]
e2_az5_data = e_all_az5[1]
e3_az5_data = e_all_az5[2]
#e1_az6_data = e_all_az6[0]
#e2_az6_data = e_all_az6[1]
#e3_az6_data = e_all_az6[2]

fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)


ax1.plot(t_data, e1_az1_data, label = f"s{n_seed}_m{m}_T{T}_az{az1}")
ax2.plot(t_data, e2_az1_data, label = f"s{n_seed}_m{m}_T{T}_az{az1}")
ax3.plot(t_data, e3_az1_data, label = f"s{n_seed}_m{m}_T{T}_az{az1}")

ax1.plot(t_data, e1_az2_data, label = f"s{n_seed}_m{m}_T{T}_az{az2}")
ax2.plot(t_data, e2_az2_data, label = f"s{n_seed}_m{m}_T{T}_az{az2}")
ax3.plot(t_data, e3_az2_data, label = f"s{n_seed}_m{m}_T{T}_az{az2}")

ax1.plot(t_data, e1_az3_data, label = f"s{n_seed}_m{m}_T{T}_az{az3}")
ax2.plot(t_data, e2_az3_data, label = f"s{n_seed}_m{m}_T{T}_az{az3}")
ax3.plot(t_data, e3_az3_data, label = f"s{n_seed}_m{m}_T{T}_az{az3}")

ax1.plot(t_data, e1_az4_data, label = f"s{n_seed}_m{m}_T{T}_az{az4}")
ax2.plot(t_data, e2_az4_data, label = f"s{n_seed}_m{m}_T{T}_az{az4}")
ax3.plot(t_data, e3_az4_data, label = f"s{n_seed}_m{m}_T{T}_az{az4}")

ax1.plot(t_data, e1_az5_data, label = f"s{n_seed}_m{m}_T{T}_az{az5}")
ax2.plot(t_data, e2_az5_data, label = f"s{n_seed}_m{m}_T{T}_az{az5}")
ax3.plot(t_data, e3_az5_data, label = f"s{n_seed}_m{m}_T{T}_az{az5}")
"""
ax1.plot(t_data, e1_az6_data, label = f"s{n_seed}_m{m}_T{T}_az{az6}")
ax2.plot(t_data, e2_az6_data, label = f"s{n_seed}_m{m}_T{T}_az{az6}")
ax3.plot(t_data, e3_az6_data, label = f"s{n_seed}_m{m}_T{T}_az{az6}")
"""

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