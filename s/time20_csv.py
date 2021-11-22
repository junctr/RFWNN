import csv
import numpy as np

t = 0
end = 20
step = 0.0001

t_data = []

while t < end:

    t_data.append(t)

    t += step

"""
with open("time20.csv","w")as f:
    dataWriter = csv.writer(f)
    dataWriter.writerow(t_data)

f.close()
"""

np.savetxt("time20.csv",t_data)