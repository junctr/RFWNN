def rungekutta(t, q, p, l, g, odot, co, ro, W):
    
    e = e_f(t,q)
    s = s_f(e)
    x = x_f(t,q,s)
    xji = xji_f(t,x,xold,odot,co,ro)
    A = A_f(t,xji,co,ro)
    Aold = A_f(t,xold, co,ro)
    B = B_f(x,Aold,odot,ro)
    mu = mu_f(A)
    muji = muji_f(A)
    omega = omega_f(odot,co,ro,W)
    tau = tau_f(s,A,beta,zeta,omega)

    return dq