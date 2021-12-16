def tau0(t):
    tau0 = np.array([
        [2*np.sin(10 *2*np.pi*t)],
        [2*np.sin(10 *2*np.pi*t)],
        [2*np.sin(10 *2*np.pi*t)]],
        dtype=np.float64
    )
    
    return tau0

