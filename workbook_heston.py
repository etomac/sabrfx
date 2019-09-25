# heston Library

from QuantLib import *
import numpy as np
from math import log, exp, pi
from black import bs, impvol
import scipy 
from pplib_sub import ad2pd
from scipy.optimize import differential_evolution

def Trapez(yA,xA):
    integral = 0.0
    for idx in range(len(xA)-1):
        xdiff = xA[idx+1] - xA[idx]
        ysum  = yA[idx+1] + yA[idx]
        piece = ysum * xdiff
        integral += piece
    return integral * 0.5
    
class hestonme:

    def __init__(self, spot, tenorann, riskfree, dividend, hestonparams, isdf=False, isfwd=False):
    
        self.pi     = pi
        self.spot   = spot
        self.t      = tenorann
        self.risk   = 0.0
        self.params = hestonparams 
        self.kappa  = self.params['mr']
        self.theta  = self.params['varLR']
        self.vnull  = self.params['varInit']
        self.sigma  = self.params['volOfVar']
        self.rho    = self.params['rho']
        
        if isdf:
            self.df     = riskfree
            self.rf     = -log(riskfree)/tenorann
        else:
            self.rf     = riskfree
            self.df     = np.exp(-riskfree*tenorann)
            
        if isfwd:
            self.fwd    = dividend
            self.div    = self.rf - log(dividend/spot)/tenorann 
        else:
            self.fwd    = spot * np.exp((self.rf-dividend)*tenorann)
            self.div    = dividend
        
        
    def charparams(self, bType, evalPt):
        uOut = -0.5
        bOut = self.risk + self.kappa
        if bType == 1:
            bOut = bOut - self.rho * self.sigma
            uOut = 0.5
        dOut = complex(-1.*bOut, self.rho*self.sigma*evalPt)**2 - self.sigma**2 * complex(-1.*evalPt**2, 2.*uOut*evalPt) 
        dOut = dOut ** 0.5
        gOut = (complex(bOut+dOut, -self.rho*self.sigma*evalPt))/(complex(bOut-dOut, -self.rho*self.sigma*evalPt))
            
        return {'g':gOut, 'd': dOut, 'b': bOut, 'u':uOut}
      
    def Cval(self, bType, evalPt):
    
        outs = self.charparams(bType, evalPt)
        bOut, dOut, gOut = outs['b'], outs['d'], outs['g']
        
        lhterm = self.t * complex(bOut + dOut, -self.rho*self.sigma*evalPt)
        rhterm = -2.*np.log((1. - gOut* np.exp(dOut * self.t))/(1.-gOut)) 
        inbracket = lhterm + rhterm

        return complex(0,(self.rf - self.div)*evalPt*self.t) + (self.kappa*self.theta)/(self.sigma**2) \
        * inbracket
      
    def Dval(self, bType, evalPt):
    
        outs = self.charparams(bType, evalPt)
        bOut, dOut, gOut = outs['b'], outs['d'], outs['g']
        
        return complex(bOut + dOut, -1.*self.rho*self.sigma*evalPt)/(self.sigma**2) * \
        (1. - np.exp(dOut * self.t))/(1. - gOut * np.exp(dOut * self.t))

    def cfunction(self, bType, evalPt):
        return np.exp(self.Cval(bType, evalPt) + self.Dval(bType, evalPt) * self.vnull + complex(0.0, evalPt*np.log(self.spot))) 
        
    def Pval(self, bType, pxStrike, lbound=1e-5, halfiter=8500):
        
        outs = {}
        
        def Intval(theu):
            val = ((self.cfunction(bType, theu) * np.exp(complex(0.,-theu*np.log(pxStrike))))/ complex(0.,theu)).real
            if (np.isnan(val)):
               val = 0.0 
            return val
                
        x0 = np.linspace(lbound, 1.-1e-10, halfiter)  
        xA = self.spot*np.array((x0.tolist() + (1./x0).tolist()[::-1]))
        
        yA, xB = [], []
        
        for idx, px in enumerate(xA.tolist()):
            yInt = Intval(px)
            if not(np.isnan(yInt)):
                yA.append(yInt)
                xB.append(px)
        
        ints = {    
                'quad': scipy.integrate.quad(Intval, 0.0, np.inf)[0],
                'simp': scipy.integrate.simps(yA, xB),
                'trap': Trapez(yA,xB)
                }
        
        for key in ints.keys():
            outs[key] = 0.5 + ints[key]/self.pi
        
        return outs
        
    def callpx(self, pxStrike, issimp=False):
        pval1 = self.Pval(1,pxStrike) 
        pval2 = self.Pval(2,pxStrike)
        
        res   = {}
        
        for key in  pval1.keys():
            res[key] = self.df * (self.fwd * pval1[key] - pxStrike * pval2[key])
        
        return res
        
    def qlcall(self, pxStrike):
        # 1. set evaluation date
        today = Date(31, 7, 2015)
        Settings.instance().evaluationDate = today
        
        # 2. set instrument ( payoff, exercise ) 
        option = EuropeanOption(PlainVanillaPayoff(Option.Call, pxStrike), EuropeanExercise(today + Period(int(self.t * 365.25), Days)))
        
        # 3. market quotes 
        u = SimpleQuote(self.spot)
        r = SimpleQuote(self.rf)
        q = SimpleQuote(self.div)
        
        # 4. market instrument
        riskFreeCurve = FlatForward(0, TARGET(), QuoteHandle(r), ActualActual())
        dividendCurve = FlatForward(0, TARGET(), QuoteHandle(q), ActualActual())
        
        process2 = HestonProcess(YieldTermStructureHandle(riskFreeCurve),
                   YieldTermStructureHandle(dividendCurve),
                   QuoteHandle(u),
                   self.vnull, self.kappa, self.theta, self.sigma, self.rho) # v0, kappa(mrev), theta(LR), sigma(volofvar), rho
                   
        model = HestonModel(process2)
        engine2  = AnalyticHestonEngine(model)
        option.setPricingEngine(engine2)
        
        return option.NPV()
        
    def modelvols(self, strikes):
        resvols = []
        for idx, strike in enumerate(strikes):
            hestoncall = self.qlcall(strike)
            try:
                qloo = qloption(1., strike, self.fwd, self.t)
                qvol = qloo.ivol(hestoncall)
                resvols.append(qvol)
            except Exception as e:
                resvols.append(0.0)
                print e.message
        return resvols

def hestoncalib(hestonobj, strikes, vols,  maxiter=30, wgts=None):
    
    if wgts == None:
        wgts = tuple([1.0 for k in strikes])
        
    def costfunction(x, strikes, vols):
        
        '''
        Default boundary conditions (Google):
            -   kappa,     theta, v0 ,   sigma,     rho,  
            - [(0.01,15), (0,1), (0,1), (0.01,1), (-1,1),]
        '''
        
        errs = 0.0
        
        #print 
        for idx, strike in enumerate(strikes):
        
            newheston  = hestonme(hestonobj.spot, hestonobj.t, hestonobj.rf, hestonobj.div, {'mr':x[0], 'varLR':x[1], 'varInit':x[2], 'volOfVar':x[3], 'rho':x[4]})
            
            hestoncall = newheston.qlcall(strike) if (2. * x[0] * x[1] > x[3]**2) else 0.0
            
            qloo       = qloption(1., strike, hestonobj.fwd, hestonobj.t)
            qloo.vol   = vols[idx]
            bscall     = qloo.npv()
            
            thiserr   = (hestoncall - bscall) ** 2 
            
            #print idx, strike, hestoncall, bscall
            
            errs += thiserr * wgts[idx]
            
        return errs
    
    res = differential_evolution(costfunction, bounds=[(0.01, 15.),(0.01, 1.0),(0.01, 1.0),(0.01, 1.0),(-0.99, 0.99)], args=(strikes, vols), maxiter=maxiter)
    x   = res.x
    print res
    resparams  = {'mr':x[0], 'varLR':x[1], 'varInit':x[2], 'volOfVar':x[3], 'rho':x[4]}
    resheston  = hestonme(hestonobj.spot, hestonobj.t, hestonobj.rf, hestonobj.div, resparams)
    return resheston  
    
def TESTcall():

    pxS  = 100.0
    pxK  = 105.0

    rtRev      = 0.15   # k 
    rtIR       = 0.00   # r (dom, brljpy = jpy rate)
    rtDiv      = 0.00   # q (for, brljpy = brl rate)
    varLR      = 0.01   # theta
    varInit    = 0.04   # v_0
    volOfVar   = 0.05   # sigma
    rho        = -.75   # rho (correlation)
    tenorann   = 2.0
    riskprem   = 0.0    # lambda, market price of volatility risk
    
    AD4 = []
    
    for pxK in (115.,):
        for rtRev in (0.50,):
            for rtIR in (0.05,):
                for rtDiv in (-0.10,):
                    for varLR in (0.5,):
                        for varInit in (0.1, 0.5):
                            for volOfVar in (0.50,):
                                for rho in (0.5,1e-8,-0.5):
                                    for tenorann in (2.0,10.,):
                                        newheston  = hestonme(pxS, tenorann, rtIR, rtDiv, {'mr':rtRev, 'varLR':varLR, 'varInit':varInit, 'volOfVar':volOfVar, 'rho':rho})
                                        newheston.risk = riskprem
                                        hestoncall = newheston.callpx(pxK)
                                        qlcall     = newheston.qlcall(pxK)
                                        bscall     = bs(1., newheston.fwd, pxK, varInit**0.5, tenorann, newheston.df)
                                        bscall2    = bs(1., newheston.fwd, pxK, varLR**0.5, tenorann, newheston.df)
                                        
                                        thisobj = {'pxK':pxK, 'rtRev':rtRev, 'rtIR':rtIR, 'rtDiv':rtDiv, 'varLR':varLR, 'varInit':varInit, 'volOfVar':volOfVar, 'rho':rho, 
                                        'tenorann':tenorann, 'hestonQL':qlcall, 'BSinit':bscall, 'BSlr':bscall2, 'fellercheck':2 * rtRev * varLR > volOfVar**2 }
                                        
                                        for key in hestoncall.keys():
                                            thisobj['hestonI' + key[0].upper()] = hestoncall[key] 
                                            
                                        AD4.append(thisobj)
                                            
    ad2pd(AD4).to_csv('research\\hestontest.csv')
    
    print 'done'

def TESTregression():

    xA = [float(i+1) for i in range(0,10)]
    yB = [1.,3.,5., 6.,8.,9., 15.,18.,19., 22.]
    
    def costfunction(x, xaxis, yaxis):
        
        errs = 0.0
        
        for idx, xval in enumerate(xaxis):
            errs += ((x[0] + x[1] * xval) - yaxis[idx]) ** 2
            
        return errs
    
    res = differential_evolution(costfunction, bounds=[(-100.,100.),(-100.,100.)], args=(xA, yB), maxiter=100)

    print res.x
    
def TESTcalib():
    
    newheston  = hestonme(65.25, 1.50, 0.973, 61.50, {'mr':0.0, 'varLR':0.0, 'varInit':0.0, 'volOfVar':0.0, 'rho':0.0}, isdf=True, isfwd=True)
    print newheston.div, newheston.rf, 
    
    strikes    = (np.linspace(.8,1.2,5) * newheston.fwd).tolist()
    vols       = [.166, .155, .149, .148, .150]
    
    newheston  = hestoncalib(newheston, strikes, vols, maxiter=100)
    
    print newheston.params, newheston.modelvols(strikes)
   
class qloption:
    def __init__(self, cp, strike, fwd, tenorann, df=1.0):
        self.cp       = cp
        self.strike   = strike
        self.fwd      = fwd
        self.t        = tenorann
        self.vol      = 0.2
        self.callpx   = 0.0
        self.option   = None
    def npv(self, callprice=None):
        today = Date(31, 7, 2015)
        Settings.instance().evaluationDate = today
        if self.cp == 1.0:
            payoff = PlainVanillaPayoff(Option.Call, self.strike)
        else:   
            payoff = PlainVanillaPayoff(Option.Put,  self.strike)
        exercisedate = EuropeanExercise(today + Period(int(self.t * 365.25), Days))
        option = EuropeanOption(payoff, exercisedate)

        S = QuoteHandle(SimpleQuote(self.fwd))
        r = YieldTermStructureHandle(FlatForward(0, TARGET(),QuoteHandle(SimpleQuote(0.)), ActualActual()))
        q = YieldTermStructureHandle(FlatForward(0, TARGET(),QuoteHandle(SimpleQuote(0.)), ActualActual()))
        sigma = BlackVolTermStructureHandle(BlackConstantVol(0, TARGET(), self.vol, ActualActual()))
        process = BlackScholesMertonProcess(S,q,r,sigma)
        if not(callprice==None):
            return option.impliedVolatility(callprice, process)
        else:
            engine      = AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)
            return option.NPV()
        
    def ivol(self, callprice):
        return self.npv(callprice) 
        
def main():
    fwd = 10000.
    stk = 10020.
    tea = 2.
    ppx = 0.4 * 0.2 * (2.0) ** 0.5 * fwd
    print ppx 
    for cp in (1., -1.):
        ivol = impvol(ppx, cp, stk, fwd, tea, 1.)
        ibs  = bs(cp, fwd, stk, ivol, tea, 1.)
        
        qloo  = qloption(cp, stk, fwd, tea)
        qvol  = qloo.ivol(ppx)
        qloo.vol = qvol 
        qbs   = qloo.npv()
        
        print 'ivol',ivol,'qvol',qvol,'ibs',ibs,'qbs',qbs
    
if __name__ == '__main__':
    main()