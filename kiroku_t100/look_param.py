import numpy as np

s = 4

e_all = np.loadtxt(f"bz_s{s}_e_all.csv")

print(e_all[0][-1])


param_all = np.load(f"bz_s{s}_param_all.npy",allow_pickle=True)

label_param = ["odot","co","ro","W","beta","zeta"]

for i in range(6):
    print(label_param[i])
    print(param_all[i])


param_all_old = np.load(f"bz_s{s}_param_all_old.npy",allow_pickle=True)

label_param = ["odot_old","co_old","ro_old","W_old","beta_old","zeta_old"]

for i in range(6):
    print(label_param[i])
    print(param_all_old[i][-1])

