import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm

def tau0(t):
    
    # tau0 = np.array([
    #     [2*np.sin(2*np.pi*t)],
    #     [2*np.sin(2*np.pi*t)],
    #     [2*np.sin(2*np.pi*t)]],
    #     dtype=np.float64
    # )
    
    # dlen = 1024 #ノイズデータのデータ長
    # mean = 0.0  #ノイズの平均値
    # std  = 1.0  #ノイズの分散
    
    tau0 = np.array([
        [2*np.sin(10 *2*np.pi*t)+np.random.normal()],
        [np.random.normal()],
        [np.random.normal()]],
        dtype=np.float64
    )
    
    return tau0

def dwn(t):
    
    dwn = 2*np.pi*np.cos(2*np.pi*t)
    
    return dwn

def main():
    
    t = 0.0
    end = 100
    step = 0.0001
    
    t_data = []
    tau0_data = [[] for i in range(3)]
    
    # t_data = [0.0]
    # wn_data = [0.0]
    
    for i in tqdm(range(int(end/step))):
        
        t_data.append(t)
        
        for j in range(3):
            
            tau0_data[j].append(tau0(t)[j][0])
        
        # wn_data.append(np.random.normal())
        # wn_data.append(wn_data[-1]-0.0001*wn_data[-1]+np.random.normal())
        # wn_data.append(wn_data[-1]+(-0.01*wn_data[-1]+np.random.normal())*step)
        # wn_data.append(wn_data[-1]+(-0.01*wn_data[-1]+np.random.normal()+dwn(t))*step)
        t += step
    
    
    
    plt.plot(t_data,tau0_data[0])
    # plt.plot(t_data,wn_data)
    
    plt.savefig(f"data/wn_step{step}.png")
    
    plt.show()
    
    return

if  __name__ == "__main__":
    
    main()