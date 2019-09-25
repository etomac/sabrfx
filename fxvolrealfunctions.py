
import datetime as dt
import numpy as np
import sys, json 
import black as blk
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as ss
import copy 
import pymongo
import time 

from math import log, exp, sqrt, pi
from scipy.stats import norm
#from sabrme import sabr, sabrfit
from interp import interp, interpd
from calcdate import calcdate, xaxis
from sabrme2 import smile
from scipy.optimize import fmin_powell, minimize, minimize_scalar, basinhopping

def createATM(fwd, time, atmv, rr, ispa, isdns):
    # {'type': 'key', 'k':fwd, 'fwd':fwd, 'vol':atmv, 'deltac':0.5, 'deltap':-0.5}
    if(not(ispa) and isdns): # npa dns (simple dns)

        # simple dns world: v4, k2
        v4 = atmv 
        k2 = fwd * exp(0.5 * v4 * v4 * time)

        # simple atmf world: d3, v3
        d3 = -norm.cdf(-0.5 * v4 * sqrt(time))
        v3 = v4 - rr/50*100 * (d3 - - 0.5)

        # premadj atmf world: v1, d1
        v1 = v3
        d1 = -norm.cdf(0.5 * v1 * sqrt(time))

        # premadj dns world: d2, v2, k1
        rrwidth = 50
        d2 = -0.5*exp(-0.5 * v4* v4 * time)
        v2 = v3 - rr/rrwidth*100 * (d2 - d1)
        k1 = fwd * exp(-0.5 * v2 * v2 * time)

    elif(not(ispa) and not(isdns)): # npa atmf
        
        # simple atmf world: d3, v3
        v3 = atmv
        d3 = -norm.cdf(-0.5 * v3 * sqrt(time))

        # simple dns world: v4, k2
        v4 = v3 + rr/50*100 * (d3 - - 0.5) 
        k2 = fwd * exp(0.5 * v4 * v4 * time) 
        
        # premadj atmf world: v1, d1
        v1 = v3
        d1 = -norm.cdf(0.5 * v1 * sqrt(time))

        # premadj dns world: d2, v2, k1
        rrwidth = 50
        d2 = -0.5*exp(-0.5 * v4* v4 * time)
        v2 = v3 - rr/rrwidth*100 * (d2 - d1)
        k1 = fwd * exp(-0.5 * v2 * v2 * time)

    elif(ispa and isdns): # pa dns
        # premadj dns world: d2, v2, k1
        v2 = atmv
        k1 = fwd * exp(-0.5 * v2 * v2 * time)
        d2 = -0.5 * exp(-0.5 * v2* v2 * time)
        # premadj atmf world: v1, d1
        rrwidth = 50
        d1 = -norm.cdf(0.5 * v2 * sqrt(time))
        v1 = v2 + rr/rrwidth*100 * (d2 - d1)
        # simple atmf world: d3, v3
        v3 = v1
        d3 = -norm.cdf(-0.5 * v3 * sqrt(time))
        # simple dns world: v4, k2
        v4 = v3 + rr/50*100 * (d3 - - 0.5) 
        k2 = fwd * exp(0.5 * v4 * v4 * time) 
    else: # pa atmf 
        # premadj atmf world: v1, d1
        v1 = atmv
        d1 = -norm.cdf(0.5 * v1 * sqrt(time))
        
        # premadj dns world: d2, v2, k1
        rrwidth = 50
        d2 = -0.5 * exp(-0.5 * v1* v1 * time)
        v2 = v1 - rr/rrwidth*100 * (d2 - d1)
        k1 = fwd * exp(-0.5 * v2 * v2 * time)

        # simple atmf world: d3, v3
        v3 = v1
        d3 = -norm.cdf(-0.5 * v3 * sqrt(time))
        # simple dns world: v4, k2
        v4 = v3 + rr/50*100 * (d3 - - 0.5) 
        k2 = fwd * exp(0.5 * v4 * v4 * time) 

    return [
        {'type': 'atmfpa',  'k':fwd, 'fwd':fwd, 'vol':v1, 'deltac':1+d1, 'deltap':d1},
        {'type': 'dnspa',   'k':k1, 'fwd':fwd, 'vol':v2, 'deltac':-1*d2, 'deltap':d2},
        {'type': 'atmfnpa', 'k':fwd, 'fwd':fwd, 'vol':v3, 'deltac':1+d3, 'deltap':d3},
        {'type': 'dnsnpa',  'k':k2, 'fwd':fwd, 'vol':v4, 'deltac':0.5, 'deltap':-0.5},
        ]

def solvevatmf(obj):
    
    vdns  = obj['dns']
    rr25  = obj['rr25']
    tenor = obj['tenor']
    fwd   = obj['fwd']
    ispa  = obj['ispa']
    
    def solveatmf(x):
    
        vatmf        = x[0]
        delta1       = rr25/50
        cp           = -1.0
        
        if ispa:
            putdeltadns  = 0.5*cp*exp(-0.5*tenor*(0.01*vdns)**2)
            putdeltaatmf = cp*norm.cdf(-0.5*cp*vatmf*sqrt(tenor))
            vtarget      = vdns - delta1 * 100 * (putdeltaatmf - putdeltadns) 
        
        else:
            putdeltadns  = 0.5*cp
            putdeltaatmf = cp*norm.cdf(cp*0.5*vatmf*sqrt(tenor))
            vtarget      = vdns - delta1 * 100 * (putdeltaatmf - putdeltadns) 
            
        
        # print putdeltadns, putdeltaatmf, vtarget, vatmf
        
        return (vtarget - vatmf) ** 2
        
    res = minimize(solveatmf, np.array([vdns]),method='nelder-mead', options={'disp':False})

    return res.x[0]
    
def gendeltastrikes():
    outcore = []
    core = np.linspace(-0.01, -0.49, 49).tolist()
    core2 = [-i for i in sorted(core)]
    core2.insert(0,0.5)
    for nb in (core + core2):
        outcore.append(round(nb*100)/100)
    return tuple(outcore)
    
# cleanvolb, changing the variable type only (e.g. float to string)

def cleanvolb(volbobj, volscale=1.0):
    
    # clean main
    volbobj['curveDate'] = str(volbobj['curveDate'])
    volbobj['spot'] = float(volbobj['spot'])
    
    # clean conventions
    volbobj['conventions']['fwdscale'] = float(volbobj['conventions']['fwdscale'])
    
    # clean df
    volbobj['dfs'][0]['settle'] = str(volbobj['dfs'][0]['settle'])
    for idx in range(len(volbobj['dfs'][0]['dfs'])):
        volbobj['dfs'][0]['dfs'][idx]['df'] = float(volbobj['dfs'][0]['dfs'][idx]['df'])
        volbobj['dfs'][0]['dfs'][idx]['expiry'] = str(volbobj['dfs'][0]['dfs'][idx]['expiry'])
    
    # clean fwdscale
    for item in volbobj['forwards']:
        item["expiry"] = str(item["expiry"])
        item["fwd"] = float(item["fwd"])
    
    # clean vols
    for item in volbobj['vols']:
        item['tenorann'] = float(item['tenorann'])
        item["expiry"] = str(item["expiry"])
        for key in item["smilemodel"]["params"].keys():
            item["smilemodel"]["params"][key] = float(item["smilemodel"]["params"][key])
        for vol in item["vol"]:
            vol["strike"] = str(vol["strike"])
            vol["vol"] = float(vol["vol"]) * volscale

    return volbobj

# dfObject (Completed)
class dfObject():
    def __init__(self, obj):
        self.logtext   = ''
        self.obj       = obj # volb Object
        self.curvedate = obj['curveDate'] # String 
        self.settledt  = dt.datetime.strptime(obj['dfs'][0]['settle'],'%Y%m%d') # String to datetime Object
        self.ccy       = obj['dfs'][0]['label'] # String
        self.dfSeries  = obj['dfs'][0]['dfs'] # Array of objects
        self.dfSDates  = [] # Array, append with expiry String
        self.dfSDF     = [] # Array, append with df float
        self.dfSTenor  = [] # Array, append with tenor float
        for item in self.dfSeries:
            self.dfSDates.append(item['expiry'])
            self.dfSDF.append(item['df'])
            self.dfSTenor.append((dt.datetime.strptime(item['expiry'],'%Y%m%d') - self.settledt).days / 365.00)
        self.zeroCont = [] # Array, append with tenor float
        for idx in range(len(self.dfSTenor)):
            self.zeroCont.append(log(self.dfSDF[idx]) * -1.0/self.dfSTenor[idx])
    def interp(self, datetext):
        # given (today's date in datetext, array of dates in datetext, array of float, interpolate datetext) return a float value
        return interpd(self.settledt.strftime('%Y%m%d'), self.dfSDates, self.zeroCont, datetext)

# fwdObject (Completed)
class fwdObject():
    def __init__(self, obj):
        self.obj       = obj
        self.curvedate = obj['curveDate'] # String
        self.ccy       = obj['label'] # String
        self.spot      = obj['spot'] # Float
        self.fSeries   = obj['forwards'] # Array of objects
        self.fDates    = [] # Array, append with expiry String 
        self.fLevels   = [] # Array, append with fwd float
        self.fTenors   = [] # Array 
        for item in self.fSeries:
            if (item['type'] == 'Point'):
                fwdNow = self.obj['conventions']['fwdscale'] * item['fwd'] + self.spot
            else:
                fwdNow = item['fwd']
            self.fDates.append(item['expiry'])
            self.fLevels.append(fwdNow)
            self.fTenors.append((dt.datetime.strptime(str(item['expiry']),'%Y%m%d') - dt.datetime.strptime(str(self.curvedate),'%Y%m%d')).days / 365.00)
    def interp(self, datetext):
        return interpd(str(self.curvedate), self.fDates, self.fLevels, datetext)
        
if __name__ == '__main__':

    ispas = (True, False)
    isdns = (True, False)

    for pa in ispas:
        for dns in isdns:
            print ''
            print 'pa',pa,'dns',dns
            obj = createATM(112.0, 2.0, 0.07, -0.02, pa, dns)
            for i in obj:
                print i