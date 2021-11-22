import csv
import numpy as np
from matplotlib import pyplot as plt

n_seed = 4
alpha_lambda = 0.0
T1 = 1
T2 = 10
T3 = 100
T4 = 1000
T5 = 500
T6 = 5000
end = 25

t_data = np.loadtxt(f"time{end}.csv")
#e_data = np.loadtxt(f"p_s{s}_e_all.csv")
#e_all = np.load(f"p_s{s}_e_all.npy",allow_pickle=True)

e_all_T1 = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T1}_t{end}_e_all.csv")
e_all_T2 = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T2}_t{end}_e_all.csv")
e_all_T3 = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T3}_t{end}_e_all.csv")
e_all_T4 = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T4}_t{end}_e_all.csv")
e_all_T5 = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T5}_t{end}_e_all.csv")
e_all_T6 = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T6}_t{end}_e_all.csv")


e1_T1_data = e_all_T1[0]
e2_T1_data = e_all_T1[1]
e3_T1_data = e_all_T1[2]
e1_T2_data = e_all_T2[0]
e2_T2_data = e_all_T2[1]
e3_T2_data = e_all_T2[2]
e1_T3_data = e_all_T3[0]
e2_T3_data = e_all_T3[1]
e3_T3_data = e_all_T3[2]
e1_T4_data = e_all_T4[0]
e2_T4_data = e_all_T4[1]
e3_T4_data = e_all_T4[2]
e1_T5_data = e_all_T5[0]
e2_T5_data = e_all_T5[1]
e3_T5_data = e_all_T5[2]
e1_T6_data = e_all_T6[0]
e2_T6_data = e_all_T6[1]
e3_T6_data = e_all_T6[2]

fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)


ax1.plot(t_data, e1_T1_data, label = f"s{n_seed}_m{alpha_lambda}_T{T1}")
ax2.plot(t_data, e2_T1_data, label = f"s{n_seed}_m{alpha_lambda}_T{T1}")
ax3.plot(t_data, e3_T1_data, label = f"s{n_seed}_m{alpha_lambda}_T{T1}")

ax1.plot(t_data, e1_T2_data, label = f"s{n_seed}_m{alpha_lambda}_T{T2}")
ax2.plot(t_data, e2_T2_data, label = f"s{n_seed}_m{alpha_lambda}_T{T2}")
ax3.plot(t_data, e3_T2_data, label = f"s{n_seed}_m{alpha_lambda}_T{T2}")

ax1.plot(t_data, e1_T3_data, label = f"s{n_seed}_m{alpha_lambda}_T{T3}")
ax2.plot(t_data, e2_T3_data, label = f"s{n_seed}_m{alpha_lambda}_T{T3}")
ax3.plot(t_data, e3_T3_data, label = f"s{n_seed}_m{alpha_lambda}_T{T3}")

ax1.plot(t_data, e1_T4_data, label = f"s{n_seed}_m{alpha_lambda}_T{T4}")
ax2.plot(t_data, e2_T4_data, label = f"s{n_seed}_m{alpha_lambda}_T{T4}")
ax3.plot(t_data, e3_T4_data, label = f"s{n_seed}_m{alpha_lambda}_T{T4}")
"""
ax1.plot(t_data, e1_T5_data, label = f"s{n_seed}_m{alpha_lambda}_T{T5}")
ax2.plot(t_data, e2_T5_data, label = f"s{n_seed}_m{alpha_lambda}_T{T5}")
ax3.plot(t_data, e3_T5_data, label = f"s{n_seed}_m{alpha_lambda}_T{T5}")

ax1.plot(t_data, e1_T6_data, label = f"s{n_seed}_m{alpha_lambda}_T{T6}")
ax2.plot(t_data, e2_T6_data, label = f"s{n_seed}_m{alpha_lambda}_T{T6}")
ax3.plot(t_data, e3_T6_data, label = f"s{n_seed}_m{alpha_lambda}_T{T6}")
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