import csv

data = []

t = 0

for i in range(10):
    data.append(t)
    t += 1

print(data)

with open("out.csv","w")as f:
    dataWriter = csv.writer(f)
    dataWriter.writerow(data)

f.close()