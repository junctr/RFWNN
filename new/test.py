import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm

tau0_data = [[0.0] for i in range(3)]

for j in range(5):
    
    for i in range(3):
        tau0_data[i].append(j)

print(tau0_data)