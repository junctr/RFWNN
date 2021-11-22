import csv
import numpy as np
from matplotlib import pyplot as plt

n_seed = 4
alpha_lambda = 0.3
az = 0.1
az1 = 0.4
T = 1000
end = 25

t_data = np.loadtxt(f"time{end}.csv")
#e_data = np.loadtxt(f"p_s{s}_e_all.csv")
#e_all = np.load(f"p_s{s}_e_all.npy",allow_pickle=True)

e_all_p = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T}_t{end}_az{az1}_e_all.csv")
e_all_c = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T}_t{end}_az{az}_e_all.csv")

e1_p_data = e_all_p[0]
e2_p_data = e_all_p[1]
e3_p_data = e_all_p[2]
e1_c_data = e_all_c[0]
e2_c_data = e_all_c[1]
e3_c_data = e_all_c[2]

fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(t_data, e1_c_data, label = "Conventional")
ax2.plot(t_data, e2_c_data, label = "Conventional")
ax3.plot(t_data, e3_c_data, label = "Conventional")

ax1.plot(t_data, e1_p_data, label = f"Proposed_az{az1}")
ax2.plot(t_data, e2_p_data, label = f"Proposed_az{az1}")
ax3.plot(t_data, e3_p_data, label = f"Proposed_az{az1}")

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