# sabrme2.py (smile class)
from scipy.stats import norm
import copy
from random import Random
import numpy as np
import black as blk
import scipy.special as ss
import matplotlib.pyplot as plt
from math import log, sqrt, pi, exp, isnan
from scipy.optimize import fmin_powell, minimize, minimize_scalar, basinhopping

# ------------------------------------------------------------------------------------------------------------------------------------------
# smile class
# ------------------------------------------------------------------------------------------------------------------------------------------ 
#
# content:
# 
# definition of terms: 
#    alpha = initial volatility
#    beta  = skewness
#    rho   = correlation
#    nu    = vol of vol 
#
# sabr         | generic sabr
# lnfit        | list of strikes AND vols (NUMPY) | return parmams lognormal fitter    (Beta = 1, Alpha = ATMF vol) 
# fit          | list of strikes AND vols (NUMPY) | return params original fitter
# thissabr     | k                                | return modelled vols               (can take self.params or otherparams)
# thisbs       | k/cp                             | return black scholes prices        (can take self.params or otherparams AND input vols)
# modelvols    | list of strikes (ARRAY)          | return model vols with 'thissabr'  (can take self.params or otherparams)
# modeldeltas  | list of strikes (ARRAY) + isadj  | generate forward deltas            (can take self.params or otherparams)
# delta2strike |
#
# ------------------------------------------------------------------------------------------------------------------------------------------

class smile():
    
    def __init__(self, fwd, t, df):
    
        self.fwd     = (fwd)
        self.t       = (t)
        self.df      = (df)
        self.params  = {}
        
    # alpha = initial vol, nu = vol of vol
    
    # atmfsabr uses atmfvol instead of alpha as input, calculate the alpha first before doing anything 
    # 2 params version - meaning atmfvol and beta input is required.
    def atmfsabr2(self, k, f, expiry, atmfvol, beta, rho, nu, extended=False):
    
        if f==k:
            return atmfvol
        else:
            # 1. solve alpha from atmf (Quadratic)
            term5 = ((2 - 3 * rho * rho) * (nu ** 2)) / 24
            qa = expiry * 0.25 * rho * nu
            qb = (1 + expiry * term5)
            qc = -1.0 * atmfvol
            #print qa, qb, qc, qb**2 - 4*qa*qc
            qroot = ((-qb - (qb**2 - 4*qa*qc)**0.5)/(2*qa) , (-qb + (qb**2 - 4*qa*qc)**0.5)/(2*qa))
            
            alpha = qroot[0] if ((abs(atmfvol - qroot[0]) < abs(atmfvol - qroot[1])) and (qroot[0] > 0)) else qroot[1]
            
            #
            # First Denominator, only uses beta
            term1 = ((1 - beta) ** 2) / 24 * (log(f / k) ** 2) # ok
            term2 = ((1 - beta) ** 4) / 1920 * (log(f / k) ** 4) # ok
            firstdeno = ((f * k) ** ((1 - beta) * 0.5)) * (1 + term1 + term2) # ok
            
            # term 3, 4 (5 above)
            term3 = (alpha * alpha * ((1 - beta) ** 2)) / (24 * (f * k) ** (1 - beta)) # ok
            term4 = 0.25 * ((rho * beta * nu * alpha) / ((f * k) ** (0.5 * (1 - beta)))) # ok
            
            # zeta ok
            zeta = nu / alpha * ((f * k) ** ((1 - beta) * 0.5)) * log(f / k)
            
            # x(zeta)
            xeta = log((sqrt(1 - 2 * rho * zeta + zeta * zeta) + zeta - rho) / (1 - rho))
            
            # final equation ok
            bunch = term3 + term4 + term5 
            vol = alpha / firstdeno * (zeta / xeta) * (1 + bunch * expiry)
            
            if extended:
                return {'truealpha': alpha, 'vol':vol, 'atmfvol': atmfvol, 'qroot':qroot}
            else:
                return vol
    
    # traditional sabr uses alpha as input
    def sabr(self, k, f, expiry, alpha, beta, rho, nu, biasCalc=False):
        
        term5 = ((2 - 3 * rho * rho) * (nu ** 2)) / 24
        
        # BIAS CALC
        # First Denominator, only uses beta
        bias = 0.0
        
        if (k==f):
            term3a = ((1 - beta) ** 2) / 24 * (alpha * alpha) / (f ** (2 - 2 * beta))
            term4a = 0.25 * ((rho * beta * nu * alpha) / (f ** (1 - beta)))
            bunch2 = term3a + term4a + term5
            atmsabr = alpha / (f ** (1 - beta)) * (1 + expiry * bunch2)
            return atmsabr - bias
        
        #elif (not(beta==1)):
        else:
            # First Denominator, only uses beta
            term1 = ((1 - beta) ** 2) / 24 * (log(f / k) ** 2) # ok
            term2 = ((1 - beta) ** 4) / 1920 * (log(f / k) ** 4) # ok
            firstdeno = ((f * k) ** ((1 - beta) * 0.5)) * (1 + term1 + term2) # ok
            
            # term 3, 4 (5 above)
            term3 = (alpha * alpha * ((1 - beta) ** 2)) / (24 * (f * k) ** (1 - beta)) # ok
            term4 = 0.25 * ((rho * beta * nu * alpha) / ((f * k) ** (0.5 * (1 - beta)))) # ok
            
            # zeta ok
            zeta = nu / alpha * ((f * k) ** ((1 - beta) * 0.5)) * log(f / k)
            
            # x(zeta)
            xeta = log((sqrt(1 - 2 * rho * zeta + zeta * zeta) + zeta - rho) / (1 - rho))
            
            # final equation ok
            bunch = term3 + term4 + term5 
            otmsabr = alpha / firstdeno * (zeta / xeta) * (1 + bunch * expiry)
            return otmsabr - bias
    
    def fit3(self, strikes, vols, params):
        # given "rho, nu, beta" and "pseudo ATM" imply the alpha
        beta = 1.0
        def vol_se(x):
            volm = np.array([self.atmfsabr2(k, self.fwd, self.t, x[0], beta, params['rho'],
                                    params['nu']) for k in strikes])
            return sum((volm - vols)**2)   
        # set initial conditions
        x0 = np.array([params['atmv']]) 
        
        # minimize
        res = minimize(vol_se, x0, method='L-BFGS-B', bounds=((params['atmv']*exp(-1), params['atmv']*exp(1)),), options={'ftol':1e-18, 'gtol':1e-10})
        
        # output
        atmfvol = res.x[0]
        alpha   = self.atmfsabr2(self.fwd * (1.0-1e-10), self.fwd, self.t, atmfvol, beta, params['rho'], params['nu'], extended=True)['truealpha']
        self.params = {'alpha':alpha, 'beta':1.0, 'rho':params['rho'], 'nu':params['nu']}
        
        return atmfvol
    
    
    def fit2(self, strikes, vols, params, iswidebounds=False, isprint=False, isonevol=False):
        
        # input atmfvol as alpha
        
        atmfvol = params['alpha']
        beta    = 1.0 
        
        if False:
            rho1vol              = -0.01
            nu1vol               = 0.01
            self.params['rho']   = rho1vol
            self.params['nu']    = nu1vol
            self.params['alpha'] = self.atmfsabr2(self.fwd * (1.0-1e-10), self.fwd, self.t, atmfvol, beta, rho1vol, nu1vol, extended=True)['truealpha']
            self.params['beta']  = 1.0
            
            return {'solvedparams':self.params, 'guessparams':{'rho':rho1vol, 'nu':nu1vol}}
        # define minimisation function 
        def vol_se(x):
            volm = np.array([self.atmfsabr2(k, self.fwd, self.t, atmfvol, beta, x[0],
                                    x[1]) for k in strikes])
            return sum((volm - vols)**2)
        
        # creating first guess:
        if iswidebounds:
            rhoguess = 0.0
            nuguess  = 1.0
            bounds = [(-0.99,0.99),(1e-3,6.00)]
        else:
            matX             = np.array([[1, log(strikes[idx]/self.fwd), log(strikes[idx]/self.fwd) ** 2] for idx in range(len(strikes))])
            qparams          = np.linalg.inv(matX.T.dot(matX)).dot(matX.T).dot(np.array(vols)).tolist()
            nuguess          = max(6*qparams[0]*qparams[2]+6*qparams[1]**2.0,1e-10)**0.5
            rhoguess         = max(min(2 * qparams[1] / nuguess, 0.99),-0.99)
            bounds = [(-0.99,0.99),(1e-3,6.00)]
            
        # set initial conditions
        x0 = np.array([rhoguess, nuguess]) 
        
        # minimize
        res = minimize(vol_se, x0, method='L-BFGS-B', bounds=bounds, options={'disp':isprint, 'ftol':1e-18, 'gtol':1e-10})
        
        # output
        self.params['rho'], self.params['nu'] = res.x
        self.params['alpha'] = self.atmfsabr2(self.fwd * (1.0-1e-10), self.fwd, self.t, atmfvol, beta, res.x[0], res.x[1], extended=True)['truealpha']
        self.params['beta']  = 1.0
        
        
        return {'solvedparams':self.params, 'guessparams':{'rho':rhoguess, 'nu':nuguess}}
    
    
    def fit(self, strikes, vols, params, isprint=False, biasCalc=True):
    
        # define objective function 
        def vol_se(x):
            volm = np.array([self.sabr(k, self.fwd, self.t, x[0], params['beta'], x[1],
                                    x[2]) for k in strikes])
            return sum((volm - vols)**2)
        
        # creating first guess:
        matX             = np.array([[1, log(strikes[idx]/self.fwd), log(strikes[idx]/self.fwd) ** 2] for idx in range(len(strikes))])
        qparams          = np.linalg.inv(matX.T.dot(matX)).dot(matX.T).dot(np.array(vols)).tolist()
        alphaguess       = [qparams]
        nuguess          = (6*qparams[0]*qparams[2]+6*qparams[1]**2.0)**0.5
        rhoguess         = 2 * qparams[1] / nuguess
    
        #print alphaguess, nuguess, rhoguess
        
        # set initial conditions
        x0 = np.array([0.01, rhoguess, nuguess]) # alpha, rho, nu
        bounds = [(0.0001, 100.00), (rhoguess - 0.1, rhoguess + 0.1), (nuguess * exp(-0.35), nuguess * exp(0.35))] # alpha, rho, nu
        
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
        res = basinhopping(vol_se, x0, minimizer_kwargs=minimizer_kwargs)
       
        self.params['alpha'], self.params['rho'], self.params['nu'] = res.x
        
        # set beta to beta anyways
        self.params['beta'] = params['beta']
        
        return self.params
   
    # given k, return modelvol
    # (is validated)
    def thissabr(self, k, otherparams={}):
        if len(otherparams.keys()) == 4:
            thisparams = (otherparams)
        else:
            thisparams = self.params
        return self.sabr(k, self.fwd, self.t, thisparams['alpha'], thisparams['beta'], 
        thisparams['rho'], thisparams['nu'])
    
    # given k return black scholes price
    # (is validated)
    def thisbs(self, cp, k, otherparams={}, overwritevol=0):
        if overwritevol > 0:
            vol = overwritevol
        elif len(otherparams.keys()) == 4:
            vol = self.thissabr(k, otherparams)
        else:
            vol = self.thissabr(k)
        return blk.bs(cp, self.fwd, k, vol, self.t, self.df)
        
    def modelvols(self, inputstrikes, otherparams={}):
        if otherparams:
            return np.array([self.thissabr(k, otherparams) for k in inputstrikes])
        else:
            return np.array([self.thissabr(k) for k in inputstrikes])
    
    def modeldeltas(self, cp, inputstrikes, otherparams={}, isadj=True):
    
        deltas = []
        ks     = inputstrikes
        vs     = self.modelvols(inputstrikes, otherparams=otherparams)
        
        for idx in range(len(ks)):
            k, vol = ks[idx], vs[idx]
            deltas.append(blk.delta(cp, self.fwd, k, vol, self.t, 1.0, isfwd=True, isadj=isadj))
        return deltas
        
    def delta2strike(self, target, otherparams={}, isprint=False):
        
        # delta2strike(delta, fwd, t, dfF, vol, isfwd=False, isadj=False)
        # the issue with delta2strike is not straight forward
        
        # the input for delta2strike can be...
        # 1. -0.49, 0.03, 0.50
        # 2. dns
        # 3. atmf
        mid = self.modelvols(np.array([self.fwd]),otherparams=otherparams).tolist()[0]
        cp  = (-1. if target <= 0. else 1.)
        chg = exp( 10. * mid * sqrt(self.t)) # uses 5 standard deviations
        bd  = (self.fwd * chg ** -1., self.fwd * chg)
        
        if isprint:
            print cp
            print chg
            print bd
            print mid
        
        def f(x):
            return (self.modeldeltas(cp, [x], otherparams=otherparams)[0] - target) ** 2
            
        return minimize_scalar(f, method="Bounded", bounds=bd).x
    
    def delta2strikenew(self, target, isadj=True):
    
        # logic
        
        # user put -0.30 delta
        # pricer use blk.delta2strike with initial vol guess to get the strike
        # we then take the strike and compute MODELVOL
        # MODELVOL likely to be different to GUESSVOL
        # we then reguess the vol until this GUESSVOL = MODELVOL
        def costFunction(guessvol):
            guessvol    = guessvol[0]
            guessstrike = blk.delta2strike(target, self.fwd, self.t, 1.0, guessvol, isfwd=True, isadj=isadj)     
            modelvol    = self.modelvols([guessstrike])[0]
            return ((guessvol - modelvol) ** 2)
        
        res = minimize(costFunction, np.array([self.params['alpha']]),method='nelder-mead', options={'disp':False} )
        
        volsoln = res.x[0]
        
        strike = blk.delta2strike(target, self.fwd, self.t, 1.0, volsoln, isfwd=True, isadj=isadj)
        
        return strike 
    
    def delta2strike2(self, targetdeltas=(-0.25, 0.25), isadj=True, isprint=True):
        # delta + convention + SABR(strike-vol) = {strike, vol, delta}
        outstrikes = []
            
        for targetdelta in targetdeltas:
            alpha  = self.params['alpha']
            fwd    = self.fwd
            expiry = self.t
            cstk   =  blk.delta2strike(targetdelta, fwd, expiry, 1.0, alpha, isadj=isadj, isfwd=False)
            
            
            # first guess
            stks = []
            cp         = (-1.0 if targetdelta < 0 else 1.0)
            volinit    = self.modelvols([cstk])
            deltainit  = blk.delta(cp, fwd, cstk, volinit, expiry, 1.0, isfwd=True, isadj=isadj)
            # if deltainit > targetdelta (not in absolute terms by the way), we always move to right
            # -24 go bigger strike, 26 go bigger strike 
            # -26 go smaller strike, 24 go smaller strike
            increment = 1.0 if deltainit > targetdelta else -1.0    
            
            for i in range(0,3000):
                stk         = cstk*(1+increment*i*0.0025)
                vol         = self.modelvols([stk])
                deltaguess  = blk.delta(cp, fwd, stk, vol, expiry, 1.0, isfwd=True, isadj=isadj)
                stks.append(stk)
                if increment*deltaguess < increment*targetdelta:
                    break
                    
            if len(stks) >= 2:
                stklist = np.linspace(min(stks[-2:]),max(stks[-2:]), 100)
                for stk in stklist:
                    vol         = self.modelvols([stk])
                    deltaguess  = blk.delta(cp, fwd, stk, vol, expiry, 1.0, isfwd=True, isadj=isadj)
                    if deltaguess<targetdelta:
                        sid    = stklist.tolist().index(stk)
                        fstk   = (stklist[sid-1] + stk) * 0.5
                        fvol   = self.modelvols([fstk])
                        fdelta = blk.delta(cp, fwd, fstk, fvol, expiry, 1.0, isfwd=True, isadj=isadj)
                        
                        outstrikes.append({'strike':fstk, 'targetdelta':targetdelta, 'vol':fvol.tolist()[0], 'delta':fdelta.tolist()[0]})
                        break
                        
            else:
                outstrikes.append({'strike':cstk, 'targetdelta':targetdelta, 'vol':volinit.tolist()[0], 'delta':deltainit.tolist()[0]})

        return outstrikes
    
    def dns(self, ispa=False):
        
        dnsvol, dnsk, dnsdelta = 0,0,0
        # pa: dns is on left of atmf
        # npa: dns is on the right of atmf
        atmfk   = self.fwd
        atmfvol = self.modelvols([self.fwd]).tolist()[0]
        
        atmfcd  = blk.delta(1.0, self.fwd, self.fwd, atmfvol, self.t, 1.0, isfwd=True, isadj=ispa)
        atmfpd  = blk.delta(-1.0, self.fwd, self.fwd, atmfvol, self.t, 1.0, isfwd=True, isadj=ispa)
        
        mult    = -1.0 if ispa else 1.0 
        sign    = -1.0 if (atmfcd + atmfpd) < 0.0 else 1.0
        steps   = np.linspace(1e-5, 0.50, 10000).tolist()
        for i in steps:
            
            kn = atmfk * (1 + mult * i)
            vn = self.modelvols([kn]).tolist()[0]
            kcd  = blk.delta(1.0, self.fwd, kn, vn, self.t, 1.0, isfwd=True, isadj=ispa)
            kpd  = blk.delta(-1.0, self.fwd, kn, vn, self.t, 1.0, isfwd=True, isadj=ispa)
            #print kn, vn, kcd, kpd
            
            thissign = -1.0 if  (kcd + kpd) < 0.0 else 1.0
            if thissign != sign :
                #print diff,kn, kcd, kpd
                dnsvol   = vn
                dnsk     = kn
                dnsdelta = kcd
                
                break
        
        #return (dnsvol, dnsstrike)
        return {'vol':dnsvol, 'strike':dnsk, 'delta':dnsdelta}
    
    
    def deltablue(self, targetdeltas=(-0.01, -0.05, -0.25, -0.4), isadj=True, isprint=False ):
   # def deltablue(self, targetdeltas=(0.01, 0.05, 0.25, 0.4), isadj=True):
        
        # the aim of this 'deltablue' is to first generate a table of strike, vol, delta
        # this will be valid for ATMF? yes.. should be 
    
        outputdict = {}
        
        for targetdelta in targetdeltas:
            outputdict[targetdelta]  = {'k':self.fwd, 'vol':self.modelvols([self.fwd]).tolist()[0]}
        checkdata  = False 
        checkdata2 = False 
       # first we calculate dns for p.a.:
        
        if isadj:
            def solvedns(k):
            
               volnow = self.modelvols(k)
               
               if isprint:
                   print 'DB1: step function dns vol'
                   print volnow
               
               return (log(self.fwd/k[0]) - 0.5 * volnow * volnow * self.t)**2
            
            if isprint:
                res = minimize(solvedns, np.array([self.fwd]), method='SLSQP', bounds=((self.fwd * 0.5, self.fwd * 2.0),), options={'disp':True})
            
            else:
                res = minimize(solvedns, np.array([self.fwd]), method='SLSQP', bounds=((self.fwd * 0.5, self.fwd * 2.0),))
                #res = minimize(solvedns, np.array([self.fwd]), method='nelder-mead')
            
            if isprint:
                print res
            
            dnsStrike = res.x
            dnsDelta  = 0.5 * exp(-0.5 * self.modelvols(dnsStrike) ** 2 * self.t)
            
        else: 
            def solvedns(k):
            
               volnow = self.modelvols(k)
               
               return (log(self.fwd/k[0]) + 0.5 * volnow * volnow * self.t)**2
            
            if isprint:
                res = minimize(solvedns, np.array([self.fwd]), method='SLSQP', bounds=((self.fwd * 0.5, self.fwd * 2.0),), options={'disp':True})
            
            else:
                res = minimize(solvedns, np.array([self.fwd]), method='SLSQP', bounds=((self.fwd * 0.5, self.fwd * 2.0),))
                #res = minimize(solvedns, np.array([self.fwd]), method='nelder-mead')
            
            dnsStrike = res.x
            dnsDelta   = 0.5
        
        if isprint:
            print 'DB2: fwd, dnsstrike, params, modelvols'
            print self.fwd
            print dnsStrike
            print self.params
            print self.modelvols(dnsStrike)
        
        atmfvol = self.modelvols([self.fwd]).tolist()[0]
        atm = {'dns':
        {'k':dnsStrike.tolist()[0], 'delta':dnsDelta, 'vol':self.modelvols(dnsStrike).tolist()[0]},
        'atmf':{
        'k':self.fwd, 
        'deltac':blk.delta(1.0, self.fwd, self.fwd, atmfvol, self.t, 1.0, isfwd=True, isadj=isadj), 
        'deltap':blk.delta(-1.0, self.fwd, self.fwd, atmfvol, self.t, 1.0, isfwd=True, isadj=isadj), 
        'vol':atmfvol}}
        
        # delta check first
        
        for idx in range(len(targetdeltas)):
            item = targetdeltas[idx]
            if abs(item) < dnsDelta:
                continue
            else:
                return {'loge':'Delta too large. Please redefine target delta matrix with absolute delta lower than DNS : ' + str(dnsDelta*100), 'res':tuple([self.fwd for i in targetdeltas]), 'exres': outputdict}
        
        for idx in range(len(targetdeltas)-1):
            item = targetdeltas[idx+1]
            itemprev = targetdeltas[idx]
            if item/abs(item) != itemprev/abs(itemprev):
                return {'loge':'Put Deltas and Call Deltas within the same array - Please keep them separated', 'res': tuple([self.fwd for i in targetdeltas]), 'exres': outputdict}
            else:
                continue
                
        isPut = False
        
        if item/abs(item) < 0:
            isPut = True
        
        # it will only compute up to DNS..
        stdev  = 8; #2.58;      # z-score at 0.995
        maxvol = (1.0, 0.6) # max vol for <2y, 2y-10y
        cutoff = 2.0        # tenor cutoff 
        iter   = 200.00     # 200 points to begin with 
        maxvol = maxvol[0] if self.t < cutoff else maxvol[1]
        P2     = sqrt(self.t) * maxvol * stdev
        putlBound  = dnsStrike*exp(-1.0*P2)
        calluBound = dnsStrike*exp(P2)
        
        # define strikes of calls and puts 
        if not(isPut):
            kRange = np.linspace(log(dnsStrike),log(calluBound),iter) # log scale, more smaller figures, more spaced out larger figures
            kRange = np.exp(kRange)
            mvolscp  = self.modelvols(kRange)
            deltascp  = blk.npdelta(1.0, self.fwd, kRange, mvolscp, self.t, 1.0, isfwd=True, isadj=isadj)
        else:
            kRange  = np.linspace(log(putlBound),log(dnsStrike),iter)
            kRange  = np.exp(kRange)
            mvolscp   = self.modelvols(kRange)
            deltascp   = blk.npdelta(-1.0, self.fwd, kRange, mvolscp, self.t, 1.0, isfwd=True, isadj=isadj)
        
        
        if checkdata:
            if not(isPut):
                print 'Call'
                print dnsStrike
                for idx in range(len(mvolscp)):
                    print [round(x* 1000)/1000 for x in (mvolscp[idx], kRange[idx], deltascp[idx], blk.delta(1.0, self.fwd, kRange[idx], mvolscp[idx], self.t, 1.0, isfwd=True, isadj=isadj))]
            else:
                print 'Put'
                print dnsStrike
                for idx in range(len(mvolscp)):
                    print [round(x* 1000)/1000 for x in (mvolscp[idx], kRange[idx], deltascp[idx], blk.delta(-1.0, self.fwd, kRange[idx], mvolscp[idx], self.t, 1.0, isfwd=True, isadj=isadj))]

        # find the local bucket
        
        def findK(k, deltatarget):
            modelvol = self.modelvols(k)
            deltanow = blk.npdelta((-1.0 if isPut else 1.0), self.fwd, k, modelvol, self.t, 1.0, isfwd=True, isadj=isadj)
            return (deltanow.tolist()[0] - deltatarget) ** 2
        
  
        try:
            if isPut:
                for targetput in targetdeltas:
                    for idx in range(len(deltascp.tolist())):
                        thisidx = idx
                        thisput = deltascp.tolist()[idx]
                        if (targetput < thisput):
                            continue 
                        else:   
                            strikeBracket = (kRange[thisidx-1], kRange[thisidx])
                            res = minimize(findK, 0.5 * np.sum(strikeBracket), method='SLSQP', args=(targetput), jac=None, bounds=(strikeBracket,), options={'disp': False, 'ftol':1e-12})
                            thisstrike = res.x.tolist()[0]
                            #targetputKey = str(round(targetput*-100)) + 'DP'
                            outputdict[targetput] =  {'k': thisstrike , 'vol': self.modelvols([thisstrike,]).tolist()[0]}

                            if checkdata2:
                                print 'put'
                                print (deltascp.tolist()[thisidx-1], mvolscp[thisidx-1], deltascp.tolist()[thisidx], mvolscp[thisidx])
                                print strikeBracket
                                print 0.5 * np.sum(strikeBracket)
                                print res.x
                            break
            else:
                # counting backwards
                for targetcall in targetdeltas:
                    for idx in range(len(deltascp.tolist())):
                        thisidx = len(deltascp.tolist()) - 1 - idx
                        thiscall = deltascp.tolist()[thisidx]
                        if (targetcall > thiscall):
                            targetcall, thiscall
                            continue 
                        else:   
                            strikeBracket = (kRange[thisidx], kRange[thisidx+1])
                            res = minimize(findK, 0.5 * np.sum(strikeBracket), method='SLSQP', args=(targetcall), jac=None, bounds=(strikeBracket,), options={'disp': False, 'ftol':1e-12})
                            thisstrike = res.x.tolist()[0]
                            targetcall = targetcall #str(round(targetcall*100)) + 'DC'
                            outputdict[targetcall] = {'k': thisstrike , 'vol': self.modelvols([thisstrike,]).tolist()[0]}
                            if checkdata2:
                                print 'call'
                                print (deltascp.tolist()[thisidx+1], mvolscp[thisidx+1], deltascp.tolist()[thisidx], mvolscp[thisidx])
                                print strikeBracket
                                print 0.5 * np.sum(strikeBracket)
                                print res.x
                            break
            
            return {'loge':'success', 'res': tuple([outputdict[target] for target in targetdeltas]), 'exres':outputdict,'atm': atm}
                        
        except Exception as e: 
            return {'loge':str(e.message),'res': tuple([self.fwd for i in targetdeltas]), 'exres': outputdict, 'atm': atm}

                        
                        
    
    # below functions are second order, not really in use... :)
    def density(self, inputstrikes, otherparams={}):
        
        shift = self.fwd * 0.00005
        density = []
        
        for k in inputstrikes:
            num = self.thisbs(1.0, k + shift, otherparams=otherparams) + self.thisbs(1.0, k - shift, otherparams=otherparams) - 2*self.thisbs(1.0, k, otherparams=otherparams) 
            den = (shift) ** 2
            density.append(num/den)
        return np.array(density)
    
    def swvar(self, inputstrikes, otherparams={}, simple=False):
    # this pricer assumes input strikes are ordered!
        sumcp = 0.0
        cps   = []
        
        for idx in range(len(inputstrikes)):
            know  = inputstrikes[idx]
            cp    = (1.0 if know > self.fwd else -1.0)
            curcp = self.thisbs(cp, know, otherparams=otherparams)
            cps.append(curcp)
            
        if simple:
            sumcp = 0.0
            
        else:
            for idx in range(len(inputstrikes)-1):
                kprev  = inputstrikes[idx]
                know   = inputstrikes[idx+1]
                cpprev = cps[idx]
                cpnow  = cps[idx+1]
                
                sumcp += 0.5 * (cpprev/(kprev**2) + cpnow/(know**2)) * (know-kprev)/self.df
            
        kvar = 2.0/self.t * (sumcp)
        self.varswap = kvar
        return kvar
        
    def atm(self):
        self.atmv = self.modelvols(np.array([self.fwd])).tolist()[0]
        return self.atmv
        
        
    # car lee seasoned (doesn't work..) 
    def seasonedvol(self, inputstrikes, otherparams={}, realisedtot=0.0):
        
        
        def kintegral(inK):
        
            sumki   = []
            zirange = (np.array(inputstrikes)/self.fwd).tolist()
            
            for idx in range(len(zirange)-1):
            
                nextz = zirange[idx+1]
                prevz = zirange[idx]
                
                npp   = 0.5 + 0.5 * sqrt(1+8.*nextz)
                npm   = 0.5 - 0.5 * sqrt(1+8.*nextz)
                ntp   = 0.5 - 1./(2.*sqrt(1.+8.*nextz))
                ntm   = 0.5 + 1./(2.*sqrt(1.+8.*nextz))
                ppp   = 0.5 + 0.5 * sqrt(1+8.*prevz)
                ppm   = 0.5 - 0.5 * sqrt(1+8.*prevz)
                ptp   = 0.5 - 1./(2.*sqrt(1.+8.*prevz))
                ptm   = 0.5 + 1./(2.*sqrt(1.+8.*prevz))
                
                nexty = exp(-nextz * realisedtot)/(inK**2*nextz**0.5) * (ntp*(inK/self.fwd)**npp + ntm *(inK/self.fwd)**npm)
                prevy = exp(-prevz * realisedtot)/(inK**2*prevz**0.5) * (ptp*(inK/self.fwd)**ppp + ptm *(inK/self.fwd)**ppm)
                
                sumki.append(.5*(nexty + prevy)*(nextz-prevz))
       
            return {'val':sum(sumki)/sqrt(3.14159265359), 'kiseries':sumki}
            
            
        sumcp = 0.0
        
        for kdx in range(len(inputstrikes) - 1):
        
            nextk = inputstrikes[kdx + 1]
            prevk = inputstrikes[kdx]
            
            kdiff = nextk - prevk
            
            nextki = kintegral(nextk)
            prevki = kintegral(prevk)
            nextki = nextki['val']
            prevki = prevki['val']
            
            if inputstrikes[kdx] < self.fwd:
                nextcp = self.thisbs(-1.,nextk) 
                prevcp = self.thisbs(-1.,prevk) 
            else:
                nextcp = self.thisbs( 1.,nextk) 
                prevcp = self.thisbs( 1.,prevk) 
                
            sumcp += 0.5*((nextcp * nextki) + (prevcp *prevki))*kdiff
        
        return {'realisedtot': sumcp}
    
    def swvol(self, inputstrikes, otherparams={}, method="cl", printfmt='2.5'):

        try:
            if method=="tb":
                atmfv    = self.modelvols([self.fwd,]).tolist()[0]
                logadj = 0.5*(self.params['nu']**2)*self.t
                logmean = log(atmfv) - logadj
                
                RANGES = []
                
                for sd in (float(printfmt),):
                    startvol = logmean - sd * (self.params['nu']*self.params['nu']*(self.t))**0.5
                    endvol   = logmean + sd * (self.params['nu']*self.params['nu']*(self.t))**0.5
                    thisppend = ['sd'+str(sd),max(1,int(100*exp(startvol))), int(100*exp(endvol))]
                    RANGES.append(thisppend)
                    
                
                RANGE = RANGES[0]
                cvxadj  = 0.0
                volsteps = np.linspace(RANGE[1],RANGE[2],2000).tolist()
                volstep  = volsteps[1] - volsteps[0]
                for volstk in volsteps:
                    volstk   = float(volstk) * 0.01
                    timdens  = norm.pdf(log(volstk), loc=logmean, scale=self.params['nu']*self.t**0.5)
                    timswvol = volstk - atmfv 
                    timswvar = (-volstk**2 + atmfv**2) / (2 * atmfv)
                    timnet   = timswvol + timswvar
                    thiscvx  = timnet * timdens * volstep * 0.01
                    cvxadj  += thiscvx
                varSwap = self.swvar(inputstrikes, otherparams=otherparams)
                return varSwap**0.5 + cvxadj 
                    

            elif method == 'cl':
                volswap = 0.0
                
                # this only work with sorted strikes
                sortedstrikes = sorted(inputstrikes)
                lowstrikes   = []
                highstrikes  = []
                
                for nbk in sortedstrikes:
                    if nbk < self.fwd:
                        lowstrikes.append(nbk)
                    elif nbk > self.fwd:
                        highstrikes.append(nbk)
                    else:
                        None
                
                lowstrikes  = lowstrikes + [(1.-1e-12) * self.fwd,]
                highstrikes = [(1.+1e-12) * self.fwd,] + highstrikes
                
                #atm
                wgtstraddle = sqrt(pi*0.5) / self.fwd * ( self.thisbs(1.0, self.fwd, otherparams=otherparams) + self.thisbs(-1.0, self.fwd, otherparams=otherparams) )
                volswap += wgtstraddle
                
                # calls
                wgtcalls = 0.0
                for kdx in range(len(highstrikes)-1):
                    kh = highstrikes[kdx+1]
                    kl = highstrikes[kdx]  
                    dk = kh - kl
                    yh = sqrt(pi/(8*self.fwd*kh**3)) * (ss.i1(log(sqrt(kh/self.fwd))) - ss.i0(log(sqrt(kh/self.fwd)))) * self.thisbs(1.0, kh, otherparams=otherparams)
                    yl = sqrt(pi/(8*self.fwd*kl**3)) * (ss.i1(log(sqrt(kl/self.fwd))) - ss.i0(log(sqrt(kl/self.fwd)))) * self.thisbs(1.0, kl, otherparams=otherparams)
                    wgtcalls += 0.5 * (yh + yl) * dk
                volswap += wgtcalls    
                
                # puts
                wgtputs = 0.0
                for kdx in range(len(lowstrikes)-1):
                    kh = lowstrikes[kdx+1]
                    kl = lowstrikes[kdx]  
                    dk = kh - kl
                    yh = sqrt(pi/(8*self.fwd*kh**3)) * (ss.i0(log(sqrt(kh/self.fwd))) - ss.i1(log(sqrt(kh/self.fwd)))) * self.thisbs(-1.0, kh, otherparams=otherparams)
                    yl = sqrt(pi/(8*self.fwd*kl**3)) * (ss.i0(log(sqrt(kl/self.fwd))) - ss.i1(log(sqrt(kl/self.fwd)))) * self.thisbs(-1.0, kl, otherparams=otherparams)
                    wgtputs += 0.5 * (yh + yl) * dk
                volswap += wgtputs    
                
                volswap = 1.0/sqrt(self.t) * (volswap) # /sqrt(self.t)
                
                self.volswap = volswap
                
                if printfmt == 'atm':
                    return 1.0/sqrt(self.t) * wgtstraddle
                elif printfmt == 'cp':
                    return 1.0/sqrt(self.t) * (wgtputs + wgtcalls)
                else:
                    return volswap
            
            elif method=="cvx":
                # only support if params alpha from self.
                varSwap = self.swvar(inputstrikes, otherparams=otherparams)
                cvx     = float(self.t)/6 * (self.params['alpha'] * self.params['alpha'] * sqrt(varSwap))
                self.volswap = sqrt(varSwap) - cvx
            else:
                self.volswap = 0.0
            return self.volswap
        except ValueError as e:
            print 'valueerror',e
            return self.modelvols([self.fwd,]).tolist()[0]
        except Exception as e:
            print e
            self.atm()
            self.volswap = self.atmv
            #print e
            return self.volswap
    def montecarlo(self, paths=10000, euler=True, timestep=1, gendist=False, logremoval=False, dispFwd=False, seed1=1, seed2=2):
    # generate variance and vol given smile
        seedvector = (187372311,204110176,129995678,6155814,22612812,61168821,21228945,146764631,94412880,117623077)
        rho, beta, nu = self.params['rho'], self.params['beta'], self.params['nu']
        ismilstein = (not(euler) and beta>=1)
        if ismilstein: print 'quasiMilstein'
        
        ra = Random()
        rb = Random()
        ra.seed(seedvector[seed1])
        rb.seed(seedvector[seed2])

        sqvars    = []
        vars      = []
        alphas2   = []
        mcfwd     = []
        yearsfrac = 252
        
        # for each path
        for i in range(int(paths)):
            
            #if i%1e7 == 0: print 'pathMillionth' + str( i )
            
            fwds     = [self.fwd,]
            vols     = [self.params['alpha'],]
            vols2    = [self.params['alpha']**2,]
            
            simfwd   = fwds[0]
            simalpha = vols[0]
            
            # for each time step
            for j in range(int(yearsfrac * self.t * timestep)):
                
                timechg  = 1.0/(yearsfrac * timestep)    
                ranvar   = ra.gauss(0,1)
                ranvar2  = rb.gauss(0,1)
                x2       = rho * ranvar + sqrt(1 - rho * rho) * ranvar2
                
                if not(ismilstein):
                    if beta < 1.0:
                        simfwd     = max(simfwd + simalpha * (simfwd ** beta) * x2 * sqrt(timechg), 1e-35)
                    elif logremoval:
                        simfwd     = simfwd * exp(simalpha * ranvar * sqrt(timechg))
                    else:
                        simfwd     = simfwd * exp(-0.5 * simalpha * simalpha * timechg + simalpha * ranvar * sqrt(timechg))
                        
                    simalpha   = simalpha * exp(-0.5*nu*nu*timechg + nu*x2*sqrt(timechg))
                else:
                    dW = ranvar * sqrt(timechg)
                    simfwd = max(simfwd + simalpha * simfwd ** beta * dW + float(beta)/2.0 * (simalpha ** 2) * (simfwd ** (2.0 * beta - 1) * (dW*dW - timechg)), 1e-35)
                    simalpha = simalpha * exp(nu * x2*sqrt(timechg) - timechg * nu * nu * 0.5)
                    
                fwds.append(simfwd)
                vols.append(simalpha)
                vols2.append(simalpha**2)

            thisvol  = 0.0   
                
            for nb in range(len(fwds)-1):
                thisvol += (log(fwds[nb+1]/fwds[nb]))**2
                
            thisvol = (thisvol * yearsfrac * timestep / (len(fwds) - 1)) ** 0.5
            
            alphas2.append(sum(vols2)/len(vols2))
            mcfwd.append(simfwd)
            sqvars.append(thisvol)
            vars.append(thisvol ** 2)
            
        distvar = {'error':[]}
        if gendist:
                    
            krange      = self.fwd * np.exp(np.linspace(-4., 4., 237) * self.params['alpha'] * self.t ** 0.5)
            
            optionprems = []    
            optionvols  = []   
            realfwd     = sum(mcfwd)/paths
            
            distvar['mcfwd'] = realfwd
            nblower          = []
            
            for k in krange:
                vo = 0.0
                
                if k >= realfwd:
                    ov = sum(map(lambda F: max(F-k, 0.0), mcfwd))/paths
                    if not(ov == 0.0):
                        try:
                            vo = blk.impvol(ov, 1.0, k, realfwd, self.t, 1.0)
                        except Exception as e:
                            distvar['error'].append([k,str(e)])

                else:
                    ov = sum(map(lambda F: max(k-F, 0.0), mcfwd))/paths
                    if not(ov == 0.0):
                        try:
                            vo = blk.impvol(ov, -1.0, k, realfwd, self.t, 1.0)
                        except Exception as e:
                            distvar['error'].append([k,str(e)])
                if isnan(ov):
                    ov = 0.0
                if isnan(vo):
                    vo = 0.0
                nblower.append(sum(map(lambda F: 1.0 if F < k else 0.0, mcfwd))/paths)
                optionprems.append(ov) 
                optionvols.append(vo)  
                distvar['vols'] = optionvols
                distvar['ks']   = krange
            
        resout = {'varswap':(sum(vars)/paths)**0.5, 'mcfwd':sum(mcfwd)/paths, 'volswap':sum(sqvars)/paths, 'minmaxfwd': [max(mcfwd), min(mcfwd)], 'integratedalpha':(sum(alphas2)/paths)**0.5, 'distvar':distvar}
        
        if dispFwd:
            resout['fwdData'] = mcfwd
        
        return resout

    def genspreads(self):
        self.ratio = sqrt(self.varswap)/self.atmv
        self.spreada = self.volswap - self.atmv
        self.spreadv = sqrt(self.varswap) - self.volswap
        
        a = 1.0/(2.0 * (self.varswap ** 0.5))
        b = -1.0
        c = self.volswap - 0.5*(self.varswap**0.5)
        
        try:
            self.be = np.array([
            (-1.0 * b - sqrt(b**2 - 4.0*a*c))/(2.0*a), 
            (-1.0 * b + sqrt(b**2 - 4.0*a*c))/(2.0*a)
            ])
        except Exception as error:
            self.be = np.array([0,0])
            # repr(error)
    
        return {'var/atm':self.ratio, 'vol-atm(bps)':self.spreada*10000., 'var-vol(bps)':self.spreadv*10000., 'be(%)':100.*self.be}
    
def main():
    ###############################
    # 2018-09-24 USDTRY 3Y analysis
    ###############################
    
    tenors = [0.1, 0.25, 1.0, 2.0, 3.0027397260273974, 5.0, 10.0]
    
    for tenor in tenors:
        atmfv = 0.25303491210937495
        dnsv  = 0.27075
        dnsk  = 10.01967038528614
        
        ks = [6.701003187857495, dnsk, 13.557321943138737]
        vs = [0.259055, dnsv, 0.361025]
        
        tfwd  = 8.975435322581346
        
        trysmile = smile(tfwd, tenor, 1.0)
        pa       = {'alpha': atmfv}
        
        ks = np.array(ks)
        vs = np.array(vs)
        print tenor
        print trysmile.fit2(ks, vs, pa)
        xaxis = [ks[0], tfwd, ks[1], ks[2]]
        k = trysmile.modelvols(xaxis)
        for i in range(len(xaxis)):
            print k[i]
        print trysmile.params
    ###############################
    # how to use
    ###############################
    # 1. create smile object
    cursmile = smile(8.975435322581346, 1.0, 0.9999856)
    ks  = [
                6.701003187857495,
                13.557321943138737,
                10.01967038528614
                ]
    vs     = [
    0.259055,
    0.361025,
    0.27075
    ]            
                
    # 2. calibrate
    ks = np.array(ks)
    vs = np.array(vs)
    pa = {'beta':1.0, 'alpha':0.25303491210937495}
    
    #print cursmile.fit(ks, vs, pa, biasCalc=True)
    
    # 3. gen swaps and atm prices
    xaxis = np.linspace(5, 15, num=10)
    #print xaxis
    #cursmile.t = 1.0
    #k = cursmile.modelvols(xaxis)
    #for i in range(len(xaxis)):
    #    print k[i]
    
    #print "varSwap, volSwap, atmf"
    #print 100*np.array([cursmile.swvar(xaxis)**0.5, cursmile.swvol(xaxis, method="cw"), cursmile.atm()])
    
    # 4. gen spreads
    #print cursmile.genspreads()
    
    # extra, plot charts, not related
    yaxis = xaxis
    plt.plot(xaxis, yaxis, 'r') #, ks, vs, 'b')
    #plt.show()
    
if __name__ == '__main__':

    main()
    
    
    