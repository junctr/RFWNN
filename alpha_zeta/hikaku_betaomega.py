import csv
import numpy as np
from matplotlib import pyplot as plt

n_seed = 22
alpha_lambda = 0.4
T = 1000
end = 25

t_data = np.loadtxt(f"time{end}.csv")
#e_data = np.loadtxt(f"p_s{s}_e_all.csv")
#e_all = np.load(f"p_s{s}_e_all.npy",allow_pickle=True)

e_all_p = np.loadtxt(f"k_s{n_seed}_m{alpha_lambda}_T{T}_t{end}_e_all.csv")
e_all_c = np.loadtxt(f"k_s{n_seed}_m0.0_T{T}_t{end}_e_all.csv")

e1_p_data = e_all_p[21]
e1_c_data = e_all_c[21]

plt.plot(t_data, e1_c_data, label = "Conventional")
plt.plot(t_data, e1_p_data, label = f"Proposed_m{alpha_lambda}")


#plt.xlim(0,20)
plt.legend()
plt.grid()

plt.show()