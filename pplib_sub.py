#pplib_marketdata_calcs

# PPLib (Price Portfolio Library)

import datetime as dt
import numpy as np
import sys, json 
import black as blk
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as ss
import pymongo
import time 
import csv
from collections import OrderedDict
from random import random as rd
#from xcorr import corrtable
from math import log, exp, sqrt, pi
from scipy.stats import norm
#from sabrme import sabr, sabrfit
from interp import interp, interpd
from calcdate import calcdate, xaxis
from sabrme2 import smile
from scipy.optimize import fmin_powell, minimize, minimize_scalar, basinhopping
from fxvolrealfunctions import createATM, solvevatmf, gendeltastrikes, cleanvolb, dfObject, fwdObject
from datetime import timedelta

# connect to database 
myclient   = pymongo.MongoClient("mongodb://localhost:27017/")
mydb       = myclient["pmt"]
ccol       = mydb["volcs"]     # c collection 
pcol       = mydb["positions"] # p collection 
bcol       = mydb["volbs"] 
dfcol      = mydb["imgdfs"] 

''' 
DATE : previous trade date
'''
def prevdate(currdt):
    ads = allDates(startday=4, startmonth=7, startyear=2015)
    return ads[ads.index(currdt)-1]     
''' 
DATE : next trade date
'''
def nextdate(currdt):
    ads = allDates(startday=4, startmonth=7, startyear=2015)
    return ads[ads.index(currdt)+1]        
''' 
DATE : generate all dates
'''
def allDates(startmonth=9, startday=3, startyear=2018, format='%Y%m%d'):
    
    allDates = []
    
    for i in range(3650):
        nowdt = dt.datetime(startyear,startmonth,startday) + timedelta(days=i)
        if nowdt >= dt.datetime.today():
            break
        elif not(nowdt.weekday() in (5,6)):
            allDates.append(nowdt.strftime(format))
    
    return allDates
    
'''
DATE : tenor to date
''' 
def tenor2d(startdate, tenor, backdate=False):
    
    lib    = {'D':1.0, 'W':7.0, 'M':30.5, 'Y':365.25}
    mult   = lib[tenor[-1]]
    bdmult = 1. if backdate else -1. 
    
    dtx    = dt.datetime.strptime(startdate, '%Y%m%d') + timedelta(days=bdmult*int(mult*float(tenor[:-1])))
    dtt    = dtx.strftime('%Y%m%d')
    alldts = allDates(startday=5, startmonth=5, startyear=2014)
    
    if dtt in alldts: 
        return dtt 
    else:
        for dtb in alldts:
            if int(dtt) < int(dtb):
                return dtb
                
''' 
DELTA : simple forward 
'''
def simpfwd(tenorann, ticker, cdt):
    #  NOT clever enough to do JPYUSD and USDJPY and AUDCHF and CHFAUD!
    tenors   = []
    fwds     = []
    try:
        for row in ccol.find({'curveDate':cdt, 'label2':'pltestc','label':ticker})[0]['recalib']['revols']:
            tenors.append(row['tenor'])
            fwds.append(row['fwd'])
        return interp(tenors, fwds, tenorann, schintp='cubic')
    except Exception as e:
        return e.message

''' 
DELTA : forward function 
'''
def fwdc2(tenorann, ticker, cdt):
    fwds   = []
    
    for i in (0,1):
        thisfx = ticker[i*3:(i+1)*3]
        
        if thisfx == 'USD':
            fwds.append(1.)
            continue
        
        isbr   = thisfx in ('GBP','EUR','AUD','NZD')
        if isbr:
            thisfxi = thisfx + 'USD' 
        else:
            thisfxi = 'USD' + thisfx
        
        thisfxl = simpfwd(tenorann, thisfxi, cdt)
            
        if (i==0) and isbr:
            fwds.append(thisfxl)
        elif(i==0) and not(isbr):
            fwds.append(1./thisfxl)
        elif(i==1) and isbr:
            fwds.append(1./thisfxl)
        else:
            fwds.append(thisfxl)
        
    return fwds[0] * fwds[1] * 1.

''' 
DELTA : simple spot 
'''
def readspot(ticker3, cdt):
    if ticker3 == 'USD':
        return 1.0
    
    thiscsv = []
    of  = open('thirdpartydata//fxc_'+ticker3.lower()+'.csv','rb')
    rf  = csv.reader(of)
    for l in rf: thiscsv.append(l)
    AD1 = []
    
    for csvrow in thiscsv[2:]:

        thisdate = dt.datetime.strptime(csvrow[7],'%d/%m/%Y')
        
        thisdict = {'cdate':thisdate.strftime('%Y%m%d')}
        
        thisdict['spot'] = spot = float(csvrow[8])
        
        AD1.append(thisdict)
        
    df = ad2pd(AD1)
    df.set_index('cdate',inplace=True)
    return df.loc[cdt]['spot']
    
''' 
DELTA : spot function  
'''
def spotc(ticker, cdt):
    
    br = ('NZD','AUD','GBP','EUR')
    tfor, tdom = ticker[:3], ticker[3:6]
    
    vfor = readspot(tfor, cdt)
    vdom = readspot(tdom, cdt)
    
    if not(tfor in br):
        vfor = 1./vfor
    if tdom in br:
        vdom = 1./vdom 
    
    return vfor * vdom 
    
''' 
DATE: days difference
'''
def ddf(prevsi, currsi): # days difference

    prevsi = dt.datetime.strptime(str(int(prevsi))[:8],'%Y%m%d')
    currsi = dt.datetime.strptime(str(int(currsi))[:8],'%Y%m%d')
    
    return (currsi - prevsi).days
    
''' 
DELTA : discount factor 
'''
def dfc(tenorann, cdt, ccy='USD.OIS'):
    tenors   = [1.0/365.25, 7.0/365.25]
    df1w     = dfcol.find({'curvedate':cdt,'name':'USD.OIS'})[0]['dfs'][0]['df']
    imp1w    = -1.0*log(df1w)/(7.0/365.25)
    df1d     = exp(-imp1w/365.25)
    dfs      = [df1d, df1w]
    cd       =  dt.datetime.strptime(cdt,'%Y%m%d')
    zcs      = [imp1w, imp1w]
    for row in dfcol.find({'curvedate':cdt,'name':'USD.OIS'})[0]['dfs'][1:]:
        expiry   =  dt.datetime.strptime(row['perfmt'],'%Y%m%d')
        thistenor = (expiry - cd).days/365.25
        tenors.append(thistenor)
        dfs.append(row['df'])
        zcs.append(-log(row['df'])/thistenor)
    intpzc = interp(tenors, zcs, tenorann, schintp='linear')
    intpdf = exp(-intpzc*tenorann)   
    return intpdf

''' 
VOLATILITY: sigma(k, t)  
'''

def volac(ticker, strike, tenorann, cdt, isparams=False):
    
    fwd = fwdc2(tenorann,ticker,cdt)
    
    # there are 3 scenarios
    # 1. i can find it 
    # 2. i can only find the flipped
    # 3. i cannot find it!!
    
    isflip   = False 
    beta     = 1.0
    tenors   = []
    params   = {'alpha':[],'rho':[],'nu':[]}
    
    inputparams = {'beta':1.0}
    cclist = ccol.find({'curveDate':cdt, 'label2':'pltestc','label':ticker})
    if cclist.count() == 0:
        cclist = ccol.find({'curveDate':cdt, 'label2':'pltestc','label':ticker[3:6] + ticker[:3]})
        isflip = True 
    if cclist.count() == 0:
        print ticker
        return 0.0
        
    # get the params for each tenor
    for row in cclist[0]['recalib']['revols']:
        tenors.append(row['tenor'])
        for key in params.keys():
            params[key].append(row[key])
            
    # cubic interpolate the params
    for key in params.keys():
        lntenors = [log(tenor) for tenor in tenors]
        srtenors = [tenor**0.5 for tenor in tenors]
        #lnparams = [log(param) for param in params[key]]
        intpparam = interp(lntenors, params[key], log(tenorann), schintp='linear')
        inputparams[key] = intpparam
        
    
    # re define params  
    tempfwd   = (1./fwd if isflip else fwd)
    thissmile = smile(tempfwd, tenorann, 1.0)
    thissmile.params = inputparams
    
    # for flipped we need to do an extra step 
    if isflip:
        atmv = thissmile.modelvols([thissmile.fwd,]).tolist()[0]
        ks =   np.concatenate((np.linspace(exp(-1), 1.0, 10), np.linspace(1.01, exp(1), 10)),axis=None) * tempfwd
        ks = ks.tolist()
        flipvs = thissmile.modelvols(ks)
        flipsmile1 = smile(fwd, tenorann, 1.0)
        flipsmile1.fit2([1.0/k for k in ks],flipvs, {'alpha':atmv, 'beta':1.0})
        thissmile = flipsmile1 
    
    if not(isparams):
    
        return thissmile.modelvols([strike]).tolist()[0]
    else:
        return thissmile.params

''' 
VOLATILITY: par strike 
'''

def varhelper(ticker, tenorann, cdt, fwdann=0, isvolswap=False, volswapmethod='cl', varswapmethod=True):
    
    isflip   = False
    beta     = 1.0
    tenors   = []
    params   = {'alpha':[],'rho':[],'nu':[], 'fwd':[]}
    
    cclist   = ccol.find({'curveDate':cdt, 'label2':'pltestc','label':ticker})
    if cclist.count() != 0:
        thisvol = cclist[0]
    else:
        thisvol = ccol.find({'curveDate':cdt, 'label2':'pltestc','label':ticker[3:6] + ticker[:3]})[0]
        isflip = True 
    
    inputparams = {'beta':1.0}
    for row in thisvol['recalib']['revols']:
        tenors.append(log(row['tenor']))
        for key in params.keys():
            params[key].append(row[key])
    
    if fwdann == 0:
        alltenors = [tenorann,]
    else:
        alltenors = [fwdann, fwdann + tenorann]
    
    smiles = []
    
    for intenor in alltenors:
        for key in params.keys():
            inputparams[key] = interp(tenors, params[key], log(intenor), schintp='cubic')
        
        thisfwd          = inputparams['fwd']
        thissmile        = smile(thisfwd, intenor, 1.0)
        thissmile.params = inputparams
       
        normstrikes  = np.linspace(0.8, 1.2, 21) * thisfwd
        normvols     = thissmile.modelvols(normstrikes)
        atmv         = normvols[normstrikes.tolist().index(thisfwd)]
        if isflip:
            flipsmile    = smile(1./thisfwd, intenor, 1.0)
            flipsmile.fit2([1.0/k for k in normstrikes],normvols, {'alpha':atmv, 'beta':1.0})
            thissmile    = flipsmile 
            thissmile.params['fwd'] = 1./thisfwd
            
        smiles.append(thissmile)
    
    if not((fwdann>0) and isvolswap):
        res = []
        for cursmile in smiles:
            nb           = 500
            
            strikeheader = (np.linspace(1e-3, 1.-1e-3, 500) * cursmile.fwd).tolist() 
            strikeRange  = strikeheader + [cursmile.fwd,] + [cursmile.fwd**2/k for k in strikeheader[::-1]]
            if isvolswap and volswapmethod=='cl':
                res.append(cursmile.swvol(strikeRange))
            elif isvolswap:
                res.append(cursmile.modelvols([cursmile.fwd,]).tolist()[0])
            elif varswapmethod:
                res.append(cursmile.swvar(strikeRange))
            else:
                res.append(cursmile.modelvols([cursmile.fwd,]).tolist()[0] ** 2)
        if len(res) > 1:
            fwdtotalvar = (res[1] * (fwdann + tenorann) -  res[0] * fwdann)
            if False:
                print fwdtotalvar/tenorann
                print fwdann
                print tenorann
                print res[1]
                print res[0]
            return fwdtotalvar/tenorann
        else:
            return res[0]
    else:        
        
        '''
        vol swap arena 
        '''
        
        if False:
            None 
            """ forward vol swap (old implementation - not in use) 
            
            fwdfwd  = smiles[1].fwd
            
            strikes = np.linspace(0.75, 1.25, 51) * fwdfwd
            
            if fwdfwd == 1.0:
                smiles0 = smile(1.0, fwdann, 1.0)
                smiles0.params = smiles[0].params
                smiles1 = smile(1.0, fwdann + tenorann, 1.0)
                smiles1.params = smiles[1].params
            else:
                smiles0, smiles1 = smiles
            fwdvols = ((smiles1.modelvols(strikes)**2 * (fwdann + tenorann) - smiles0.modelvols(strikes)**2 * fwdann)/(tenorann))**0.5
            
            fwdsmile = smile(fwdfwd, tenorann, 1.0)
            
            atmv     = fwdvols[strikes.tolist().index(fwdfwd)]
            fwdsmile.fit2(strikes, fwdvols, {'alpha':atmv, 'beta':1.0})
            
            strikeheader = (np.linspace(1e-3, 1.-1e-3, 500) * fwdsmile.fwd).tolist() 
            strikeRange  = strikeheader + [fwdsmile.fwd,] + [fwdsmile.fwd**2/k for k in strikeheader[::-1]]
            
            return ('fwdvar', fwdsmile.swvar(strikeRange)**.5, 'fwdvol', fwdsmile.swvol(strikeRange), 'nu', round(fwdsmile.params['nu']*1000)/1000,
            'rho', round(fwdsmile.params['rho']*1000)/1000)
            """
        else:
            res = []
            for cursmile in smiles:
                if volswapmethod == 'cl':
                    nb           = 500
                    strikeheader = (np.linspace(1e-3, 1.-1e-3, 500) * cursmile.fwd).tolist() 
                    strikeRange  = strikeheader + [cursmile.fwd,] + [cursmile.fwd**2/k for k in strikeheader[::-1]]
                    res.append(cursmile.swvol(strikeRange))
                else:
                    res.append(cursmile.modelvols([cursmile.fwd,]).tolist()[0])
            
            fwdtotalvar = (res[1] * (fwdann + tenorann) -  res[0] * fwdann)
            bbgN = fwdann * 252.
            
            extendedobj = {'res':res, 
            'bm': (1.-1./(4.*bbgN)) * ((252./bbgN) * ((fwdann + tenorann)*res[1]**2 - (fwdann)*res[0]**2)) ** 0.5,
            'sm': fwdtotalvar/tenorann}
            
            return extendedobj['sm']
''' 
VOLATILITY : realised vol
'''

def realisedvol2(ticker, startdate, enddate, extended=False):

    """ Return realised volatility (float)
    
    ticker, start date, end date to compute realised vol
    
    
    """
    
    timeseries = []
    
    flipcount  = 0
    
    for tpart in (ticker[:3], ticker[3:6]):
        if tpart == 'USD':
            continue
        else:
            thiscsv  = []
            of  = open('thirdpartydata//fxc_'+tpart+'.csv','rb')
            rf  = csv.reader(of)
            for line in rf: thiscsv.append([line[7],line[8]])
            thiscsv2 = []
            for row in thiscsv[5:]:
                if len(row[0])<8:
                    continue
                else:
                    dtobj = dt.datetime.strptime(row[0],'%d/%m/%Y')
                    intdt = int(dtobj.strftime('%Y%m%d'))
                    if not((intdt < int(startdate)) or (intdt > int(enddate))) and (dtobj.weekday() < 5):
                        thiscsv2.append(row)
            timeseries.append(thiscsv2)
            if tpart in ('GBP','AUD','EUR','NZD'):
                flipcount += 1
  
    mainseries = []
    totalvar   = []
    totallgc   = []
    if len(timeseries) == 2: 
        prevfx = 0.0
        for jdx in range(len(timeseries[0])):
            if flipcount == 0:
                fxSpot = float(timeseries[1][jdx][1])/float(timeseries[0][jdx][1])
            elif flipcount == 1:
                fxSpot = float(timeseries[0][jdx][1]) * float(timeseries[1][jdx][1])
            else:
                fxSpot =  float(timeseries[0][jdx][1]) / float(timeseries[1][jdx][1])
            if not(prevfx == fxSpot):
                mainseries.append([timeseries[0][jdx][0],fxSpot])
                
            
            if jdx != 0:
                lgc = log(fxSpot/prevfx)
                totalvar.append(lgc**2)
                totallgc.append(lgc)
                
            prevfx = fxSpot
    else:
        prevfx = 0.0
        for jdx in range(len(timeseries[0])):
            fxSpot = float(timeseries[0][jdx][1])
            if not(prevfx == fxSpot):
                
                mainseries.append([timeseries[0][jdx][0],fxSpot])
            
            if jdx != 0:
                lgc = log(fxSpot/prevfx)
                totalvar.append(lgc**2)
                totallgc.append(lgc)
                
            prevfx = fxSpot
    
    variance = (252.*sum(totalvar)/(len(mainseries)-1))
    
    if not(extended):
        return variance ** 0.5
    else:
        return {'var':variance, 'nbChg':len(mainseries)-1, 'vol':variance ** 0.5, 'totvar':totalvar, 'totlgc':totallgc, 'totspot':mainseries}
 
def realisedcorr(ticker, startdate, enddate):
    
    inv = 1. 
    if ticker[:3] in ('GBP','EUR','NZD','AUD'):
        inv = -1.
    
    ccy1 = realisedvol2('USD' + ticker[0:3], startdate, enddate, extended=True)['totlgc']
    ccy2 = realisedvol2('USD' + ticker[3:6], startdate, enddate, extended=True)['totlgc']
    
    if len(ccy1) != len(ccy2):
        print 'ERRRORORRR'
    
    cnum = (len(ccy1) * sum(np.array(ccy1) * np.array(ccy2)) - sum(np.array(ccy1)) * sum(np.array(ccy2)))
    cde1 = (len(ccy1) * sum(np.array(ccy1) * np.array(ccy1)) - sum(np.array(ccy1)) * sum(np.array(ccy1)))
    cde2 = (len(ccy2) * sum(np.array(ccy2) * np.array(ccy2)) - sum(np.array(ccy2)) * sum(np.array(ccy2)))
    
    return cnum/sqrt(cde1 * cde2) * inv

def impliedcorr(ticker, tenorann, cdt):
    
    fwd1 = fwdc2(tenorann, 'USD' + ticker[0:3], cdt)
    fwd2 = fwdc2(tenorann, 'USD' + ticker[3:6], cdt)
    fwd3 = fwdc2(tenorann, ticker, cdt)
    vol1 = volac('USD' + ticker[0:3], fwd1, tenorann, cdt)
    vol2 = volac('USD' + ticker[3:6], fwd2, tenorann, cdt)
    vol3 = volac(ticker, fwd3, tenorann, cdt)
    
    return (vol1**2 + vol2**2 - vol3**2)/(2 * vol1 * vol2)
    
def realisedvol(ticker, startdate, enddate, extended=False): 
    return realisedvol2(ticker, startdate, enddate, extended=False)
    
def realisedvolold(ticker, startdate, enddate, cleanstale=True, extended=False):
    """ Return realised volatility (float)
    
    ticker, start date, end date to compute realised vol
    """
    notvalid = int(str(enddate)[:8]) < int(str(startdate)[:8])
    if notvalid:
        return 0.0
    else:
        fxs = []
        for idx in range(1000):
            dtnow = dt.datetime.strptime(startdate, '%Y%m%d') + timedelta(days=idx)
            txnow = dtnow.strftime('%Y%m%d')
            if int(txnow) > int(str(enddate)[:8]):
                break
            if dtnow.weekday() < 5:
                spotnow = spotc(ticker, txnow)
                if idx > 0:
                    fxs.append(spotnow)
                else:
                    fxs.append(spotnow)
                    
        if cleanstale:
            fxs1 = []
            for nbx in range(len(fxs)-1):
                if fxs[nbx] == fxs[nbx+1]:
                    continue
                else:
                    fxs1.append(fxs[nbx])
            fxs = fxs1
            
        variance = 0.0
        mean     = 0.0
        for fdx in range(len(fxs)-1):
            variance += log(fxs[fdx+1]/fxs[fdx])**2
            mean     += log(fxs[fdx+1]/fxs[fdx])
        
        variance /= (len(fxs) - 1)
        mean     /= (len(fxs) - 1)
        
        if False:
            variance -= mean**2
        
        variance *= 252
        if not(extended):
            return variance ** 0.5
        else:
            return {'var':variance, 'nbChg':len(fxs)-1, 'vol':variance ** 0.5}

# func :: turn an array of dictionary to a Pandas DataFrame Object        
def ad2pd(ad, index=None):
    all = OrderedDict()
    if index == None :
        index = range(len(ad))
    for idx in range(len(ad)):
        
        if idx == 0:
            for key in ad[idx].keys():
               all[key] = [ad[idx][key],]
        else:
            for key in ad[idx].keys():
                try:
                   all[key].append(ad[idx][key])
                except:
                   all[key].append('=NA()')
    
    df = pd.DataFrame(all, index=index)

    return df
    
def allTickers(mode=0):    
    skippedtickers = [
    'TRYJPY',#
    'RUBJPY',#
    'MXNJPY',#
    'ZARJPY',#
    'BRLJPY',#
    'CNHJPY',#
    'CHFBRL',#
    'GBPBRL',
    'EURBRL',#
    'GBPMXN',
    'EURMXN',#
    'EURRUB',#
    'CHFMXN',#
    'CHFTRY',#
    'PHPJPY',#
    'JPYIDR',#
    'CNHMXN',#
    'INRJPY',#
    'SGDJPY',#
    'TWDJPY',#
    'USDINR',#
    'USDIDR',#
    'USDPHP',#
    ]

    oktickers = [
    'USDJPY',
    'USDTRY',
    'USDBRL',
    'USDZAR',
    'USDMXN',
    'USDRUB',
    'USDCNH',
    'GBPUSD',
    'AUDUSD',
    'EURUSD',
    'NZDUSD',
    'USDTWD',
    'USDCAD',
    'USDKRW',
    'USDSGD',
    'USDCHF',
    'EURJPY',
    'GBPJPY',
    'CHFJPY',
    'AUDJPY',
    'CADJPY',
    'NZDJPY',
    'GBPCHF',
    ]

    ccydict = {'USDHKD':[], 'AUDMXN':[], 'EURTRY':[], 'EURAUD':[]}
    if mode == 0:
        return oktickers
    elif mode == 1:
        return skippedtickers + oktickers
    else:
        return ccydict.keys() + oktickers + skippedtickers
        
# PRINTING FUNCTIONS -------------------------------------------------------------------------------------------------------        

# func :: strips so far
def sfstrip(holding, inputdt, exclToday=False):
    outstrips = []
    
    for strip in holding['strips']:
        
        stripdt = int(strip['date'])
        
        if exclToday and (stripdt < int(inputdt)):
            outstrips.append(strip)
        elif not(exclToday) and (stripdt <= int(inputdt)):
            outstrips.append(strip)
        else:
            None
    return outstrips

# func :: print a holding (e.g. dictionary within 'opt'), optional: printstrip
def printhd(holding, printstrip=True, laststrip=False):
    
    im2ihkeys = {'vSWP':'vswp','VSWP':'vswp','COt':'opt','FWD':'fwd'}
    
    fldshs = {
    'opt': ('id','expiry','curr','usym','qty','cp','stk','extid'),
    'fwd': ('id','expiry','curr','usym','qty','expiry'),
    'vswp':('id','expiry','curr','usym','qty','stk','expiry','smccy','type','vegaNotional','firstdate','lastdate'), #'varianceUnits',
    }

    fldsh = fldshs[im2ihkeys[holding['type']]]
    
    toprint = ''
    
    for flds in fldsh:
        toprint += str(holding[flds]) + ','
    
    if holding['type'] == 'VSWP':
        toprint += str(holding['varianceUnits']) + ','
    
    print toprint 
    if printstrip:
        strips2print = holding['strips']
        if laststrip:
            strips2print = holding['strips'][-1:]
        for strip in strips2print:
            toprint2 = ' t | '
            for fld in ('date','settledt','comment','qty','totaldollarvalue','cost','Ord Currency','type'):
                toprint2 += str(strip[fld]) + ','
            print toprint2

# func :: list all trades given filter
def listoftrades(filters, customdate='20190329', product='opt'):
    
    '''
    input:
    filters = {
    'usym':['JPYTRY','TRYJPY','USDTRY','TRYUSD'],
    'lasttradedate':['20180822']
    }
    
    give you list of trades:
    '''
    
    myclient   = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb       = myclient["pmt"]
    ccol       = mydb["volcs"] # c collection 
    pcol       = mydb["positions"] # p collection 

    obj = pcol.find({'date':customdate})[0]['dataset'][product]
    #'codename':'pltest_mar19',

    def filterf(x, filters):
        tf = True 
        for filterk in filters.keys():
            thiskeytf = False
            for subk in (filters[filterk]):
                thiskeytf = thiskeytf or x[filterk] == subk
            tf = tf and thiskeytf
        return tf
        
    return  list(filter(lambda x : filterf(x, filters), obj))
