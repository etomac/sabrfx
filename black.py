from scipy.stats import norm
from scipy.optimize import brentq, minimize
from math import log, exp, sqrt
import numpy as np
import pandas as pd

############ delta of an option ##################
##
## cp = 1. if call, -1. if put
##
## spot delta:
##      cp * exp(-rf * t) * N(cp * d1)
##
## forward delta:
##      cp * N (cp * d1)
##
## adjusted spot delta:
##      cp * exp(-rf * t) * K/f * N(cp * d2)
##
## adjusted forward delta:
##      cp * K/f * N(cp * d2)
##
## logic: cp, S/F, K, t, r/dfF, delta/vol/optpx
##
##################################################

# New! DN Strike

def dnstrike(vol, t, f=1.):
    return f * exp(-.5 * vol * vol * t)
    


# vol+strike -> price
def bs(cp, fwd, stk, vol, t, df):

    d1 = (log(fwd/stk) + 0.5 * vol * vol * t)/(vol * sqrt(t))
    d2 = d1 - vol * sqrt(t) 
    
    d1 *= cp
    d2 *= cp
    
    return cp * df * (fwd * norm.cdf(d1) - stk * norm.cdf(d2))
    
def npbs(cp, fwd, stk, vol, t, df):

    d1 = (np.log(fwd/stk) + 0.5 * vol * vol * t)/(vol * sqrt(t))
    d2 = d1 - vol * sqrt(t) 
    
    d1 *= cp
    d2 *= cp
    
    return cp * df * (fwd * norm.cdf(d1) - stk * norm.cdf(d2))
    
# new! bs greek
def bsgreek(cp, fwd, stk, vol, t, df, type='gamma'):
    
    # gamma, theta, vega, volgamma only
    if type == 'theta':
        return bs(cp, fwd, stk, vol, t - 1./365., df) - bs(cp, fwd, stk, vol, t, df) 
    elif type == 'vega':
        return bs(cp, fwd, stk, vol + 0.01, t, df) - bs(cp, fwd, stk, vol, t, df) 
    elif type == 'volgamma':
        return (bs(cp, fwd, stk, vol + 0.01, t, df) + bs(cp, fwd, stk, vol - 0.01, t, df) - 2* bs(cp, fwd, stk, vol, t, df))
    elif type == 'gamma':
        return (bs(cp, fwd*1.01, stk, vol, t, df) + bs(cp, fwd*0.99, stk, vol, t, df)     - 2* bs(cp, fwd, stk, vol, t, df)) #/(.01 * fwd)
    elif type == 'delta':
        return (bs(cp, fwd*1.01, stk, vol, t, df) - bs(cp, fwd, stk, vol, t, df)) #/(.01 * fwd)
    else:
        return 0.0
        
# price+vol+strike -> delta
def delta(cp, fwd, stk, vol, t, dfF, isfwd=False, isadj=False):    

    d1 = (log(fwd/stk) + 0.5 * vol * vol * t)/(vol * sqrt(t))
    d2 = d1 - vol * sqrt(t) 
    
    if isadj:
        delta = cp * stk/fwd * norm.cdf(d2 * cp)
    else:
        delta = cp * norm.cdf(d1 * cp)
    
    if not(isfwd):
        delta *= dfF
    
    return delta
    
    
def npdelta(cp, fwd, stk, vol, t, dfF, isfwd=False, isadj=False):    

    d1 = (np.log(fwd/stk) + 0.5 * vol * vol * t)/(vol * sqrt(t))
    d2 = d1 - vol * sqrt(t) 
    
    if isadj:
        delta = cp * stk/fwd * norm.cdf(d2 * cp)
    else:
        delta = cp * norm.cdf(d1 * cp)
    
    if not(isfwd):
        delta *= dfF
    
    return delta
    
# delta+vol -> strike
def delta2strike(delta, fwd, t, dfF, vol, isfwd=False, isadj=False): # seems tested!..

    cp    = -1. if delta < 0.0 else 1.
    scale = 1.0 if isfwd else dfF
    nonadjstrike = fwd * exp(-1*cp*norm.ppf(scale*delta*cp)*vol*sqrt(t) + 0.5*t*vol**2)
    
    if isadj:
       
        # define cost function for K       
        try:
            def findK(k):
                k = k[0]
                return (cp * k/fwd * norm.cdf(cp * (log(fwd/k) - 0.5 * vol * vol * t)/(vol * sqrt(t))) - delta) ** 2.0
                
            if cp == 1.0:
                def findKmin(k):
                    k = k[0]
                    dTwo = (log(fwd/k) - 0.5*vol*vol*t)/(vol*t**0.5)
                    return (vol * sqrt(t) * norm.cdf(dTwo) - norm.pdf(dTwo)) ** 2.0
                
                minStrike = minimize(findKmin, np.array([fwd]), method='SLSQP', jac=None, bounds=((1e-8,nonadjstrike),),options={'disp': False, 'ftol':1e-12}).x[0]
                
                boundscp = ((minStrike, nonadjstrike),)
                
                
            else: 
                boundscp = ((1e-8, nonadjstrike),)
            
            paStrike = minimize(findK, np.array([nonadjstrike]), method='SLSQP', jac=None, bounds=boundscp, options={'disp': False, 'ftol':1e-12}).x[0]
  
            return paStrike
        
        except:
            return 404
        
    else: 
        return nonadjstrike
   
# price+strike -> vol
def impvol(target_value, cp, K, fwd, t, df, sigma=0.5, itm=False):

    MAX_ITERATIONS = 1000
    PRECISION = 1.0e-10
    
    # itm!
    if itm:
        if (cp>0 and (fwd > K)) or (cp<0 and (fwd < K)):
            target_value = (K-fwd)*cp + target_value
            cp = cp * -1.0
    
    for i in xrange(0, MAX_ITERATIONS):
        price =  bs(cp, fwd, K, sigma, t, df)
        vega = bs_vega(K, fwd, sigma, t, df)

        price = price
        diff = target_value - price 

        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega
        
    return sigma
    
# vega
def bs_vega(K, fwd, vol, t, df): #vega DF not spot on!
    n = norm.pdf
    d1 = (log(fwd/K)+(vol*vol/2.)*t)/(vol*sqrt(t))
    return df * fwd * sqrt(t)*n(d1)

    
#         
# LEGACY d2delta (pa)
def deltafwd(K, fwd, vol, t, target, cp):
    return cp * K/fwd * norm.cdf(cp * (log(fwd/K) - 0.5 * vol * vol * t)/(vol * sqrt(t))) - target
    
#         
# LEGACY d2delta (pa)
def deltaspot(K, fwd, vol, t, target, cp, df):
    return df * cp * K/fwd * norm.cdf(cp * (log(fwd/K) - 0.5 * vol * vol * t)/(vol * sqrt(t))) - target
    
   
# LEGACY vol+delta -> strike
def getfstrike(fwd, vol, t, delta, cp, thres=(0,0)):
    if thres==(0,0):
        thres=(fwd*0.85, fwd*1.40)
    try:
        return brentq(deltafwd, thres[0], thres[1], args=(fwd, vol, t, delta, cp))
    except:
        return 404.00
        
# LEGACY vol+delta -> strike  
def getsstrike(fwd, vol, t, delta, cp, dfF, thres=(0,0)):
    if thres==(0,0):
        thres=(fwd*0.85, fwd*1.40)
    try:
        return brentq(deltaspot, thres[0], thres[1], args=(fwd, vol, t, delta, cp, dfF))
    except:
        return 403.00
            
# LEGACY
def mainold():
    print deltafwd(3.0, 1.2000, 0.2, 1.0, 0.0, 1.0)
    for k in [0.1 + i*0.01 for i in range(200)]:
        print deltafwd(k, 1.2, 0.2, 1.0, 0.0, -1)
    print brentq(deltafwd, 1.2*0.1, 1.2*2.0, args=(1.2, 0.2, 1.0, 0.25, 1.0))
    
def main():
    #(delta, fwd, t, dfF, vol, isfwd=False, isadj=False)
    
    fwd = 30.5
    stk = 27.9 # put
    vol = 0.2
    t   = 12.1
    dfF = 1.0
    
    print delta2strike(0.2, 30.5, t, 1.0, vol, isfwd=False, isadj=False)
    print delta2strike(0.2, 30.5, t, 1.0, vol, isfwd=False, isadj=True)
    print delta2strike(0.7946296549, 30.5, t, 1.0, vol, isfwd=False, isadj=False)
    print delta2strike(0.194975206, 30.5, t, 1.0, vol, isfwd=False, isadj=True)
    
    print delta(-1, fwd, stk, vol, t, dfF, isfwd=False, isadj=True) 
    print delta(-1, fwd, stk, vol, t, dfF, isfwd=False, isadj=False)  
    print delta(1, fwd, 33.1, vol, t, dfF, isfwd=False, isadj=True)  
    print delta(1, fwd, 33.1, vol, t, dfF, isfwd=False, isadj=False)  
    
if __name__ == "__main__":
    main()