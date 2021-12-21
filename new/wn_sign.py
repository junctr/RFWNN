import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm

alpha_wn0 = 100
alpha_wn1 = 10

def dwn_f(wn):
    
    # dwn = np.array([
    #     [0.0],
    #     [0.0],
    #     [0.0]],
    #     dtype=np.float64
    # )
    
    wnv = np.array([
        [np.random.normal()],
        [np.random.normal()],
        [np.random.normal()]],
        dtype=np.float64
    )
    
    dwn = -np.linalg.inv(alpha_wn0 * np.identity(3, dtype=np.float64))@wn + alpha_wn1 * np.identity(3, dtype=np.float64) @ wnv
    
    return dwn

def taus1_f(wn):
    
    taus1 = np.array([[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,2.0]],dtype=np.float64) @ np.sign(wn)
    
    return taus1

def main():
    
    t = 0.0
    end = 10
    step = 0.0001
    
    t_data = []
    wn_data = [[] for i in range(3)]
    taus1_data = [[] for i in range(3)]
    
    wn = np.array([
        [0.0],
        [0.0],
        [0.0]],
        dtype=np.float64
    )
    
    # t_data = [0.0]
    # wn_data = [0.0]
    
    for i in tqdm(range(int(end/step))):
        
        t_data.append(t)
        
        for j in range(3):
            
            taus1_data[j].append(taus1_f(wn)[j][0])
            wn_data[j].append(wn[j][0])
        
        wn += step * dwn_f(wn)
        t += step
    
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False)

    for i in range(3):
        
        axes[i].plot(t_data, wn_data[i])
        axes[i].plot(t_data, taus1_data[i])
        axes[i].grid()
    
    plt.show()
    
    return

if  __name__ == "__main__":
    
    main()