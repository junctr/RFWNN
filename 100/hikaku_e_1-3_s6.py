import csv
import numpy as np
from matplotlib import pyplot as plt

s1 = 4
s2 = 8
s3 = 7
s4 = 22
s5 = 12
s6 = 13

t_data = np.loadtxt("time20.csv")
e1_s1_data = np.loadtxt(f"s{s1}_e1.csv")
e2_s1_data = np.loadtxt(f"s{s1}_e2.csv")
e3_s1_data = np.loadtxt(f"s{s1}_e3.csv")
s_s1_data = np.loadtxt(f"s{s1}_s.csv")
e1_s2_data = np.loadtxt(f"s{s2}_e1.csv")
e2_s2_data = np.loadtxt(f"s{s2}_e2.csv")
e3_s2_data = np.loadtxt(f"s{s2}_e3.csv")
s_s2_data = np.loadtxt(f"s{s2}_s.csv")
e1_s3_data = np.loadtxt(f"s{s3}_e1.csv")
e2_s3_data = np.loadtxt(f"s{s3}_e2.csv")
e3_s3_data = np.loadtxt(f"s{s3}_e3.csv")
s_s3_data = np.loadtxt(f"s{s3}_s.csv")
e1_s4_data = np.loadtxt(f"s{s4}_e1.csv")
e2_s4_data = np.loadtxt(f"s{s4}_e2.csv")
e3_s4_data = np.loadtxt(f"s{s4}_e3.csv")
s_s4_data = np.loadtxt(f"s{s4}_s.csv")
e1_s5_data = np.loadtxt(f"s{s5}_e1.csv")
e2_s5_data = np.loadtxt(f"s{s5}_e2.csv")
e3_s5_data = np.loadtxt(f"s{s5}_e3.csv")
s_s5_data = np.loadtxt(f"s{s5}_s.csv")
e1_s6_data = np.loadtxt(f"s{s6}_e1.csv")
e2_s6_data = np.loadtxt(f"s{s6}_e2.csv")
e3_s6_data = np.loadtxt(f"s{s6}_e3.csv")
s_s6_data = np.loadtxt(f"s{s6}_s.csv")

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.plot(t_data, e1_s1_data, label = f"s{s1}")
ax2.plot(t_data, e2_s1_data, label = f"s{s1}")
ax3.plot(t_data, e3_s1_data, label = f"s{s1}")
ax4.plot(t_data, s_s1_data, label = f"s{s1}")
ax1.plot(t_data, e1_s2_data, label = f"s{s2}")
ax2.plot(t_data, e2_s2_data, label = f"s{s2}")
ax3.plot(t_data, e3_s2_data, label = f"s{s2}")
ax4.plot(t_data, s_s2_data, label = f"s{s2}")
ax1.plot(t_data, e1_s3_data, label = f"s{s3}")
ax2.plot(t_data, e2_s3_data, label = f"s{s3}")
ax3.plot(t_data, e3_s3_data, label = f"s{s3}")
ax4.plot(t_data, s_s3_data, label = f"s{s3}")
ax1.plot(t_data, e1_s4_data, label = f"s{s4}")
ax2.plot(t_data, e2_s4_data, label = f"s{s4}")
ax3.plot(t_data, e3_s4_data, label = f"s{s4}")
ax4.plot(t_data, s_s4_data, label = f"s{s4}")
ax1.plot(t_data, e1_s5_data, label = f"s{s5}")
ax2.plot(t_data, e2_s5_data, label = f"s{s5}")
ax3.plot(t_data, e3_s5_data, label = f"s{s5}")
ax4.plot(t_data, s_s5_data, label = f"s{s5}")
ax1.plot(t_data, e1_s6_data, label = f"s{s6}")
ax2.plot(t_data, e2_s6_data, label = f"s{s6}")
ax3.plot(t_data, e3_s6_data, label = f"s{s6}")
ax4.plot(t_data, s_s6_data, label = f"s{s6}")

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


plt.title(f"seed{s1}_{s2}_{s3}_{s4}_{s5}_{s6}")

plt.show()