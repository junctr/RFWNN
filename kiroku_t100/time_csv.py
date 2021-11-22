import csv
import time
import numpy as np

t = 0
end = 100
step = 0.0001
i = 0
t_data = []

start = time.time()

while t < end:

    if i%100 == 0:

        t_data.append(t)

        print(round(100*t/end, 1),"%",i,round(time.time()-start, 1),"s")

    t += step
    i += 1

"""
with open("time20.csv","w")as f:
    dataWriter = csv.writer(f)
    dataWriter.writerow(t_data)

f.close()
"""
print(len(t_data))
np.savetxt(f"time{end}.csv",t_data)