''' PPLib (Price Portfolio Library) '''

import datetime as dt
import numpy as np
import sys, json 
import black as blk
import pandas as pd
import pymongo
import time 
import csv

from datetime import timedelta
from random import random as rd
from math import log, exp, sqrt, pi
from scipy.stats import norm
from interp import interp, interpd
from sabrme2 import smile
from pplib_sub import tenor2d, realisedcorr, impliedcorr, listoftrades, printhd, sfstrip, allTickers, ad2pd, volac, dfc, ddf, spotc, readspot, fwdc2, simpfwd, allDates, realisedvol, realisedvol2, varhelper, prevdate, nextdate

"""''' not in used '''

import matplotlib.pyplot as plt
import scipy.special as ss
from xcorr import corrtable
from sabrme import sabr, sabrfit
from calcdate import calcdate, xaxis
from fxvolrealfunctions import createATM, solvevatmf, gendeltastrikes, cleanvolb, dfObject, fwdObject
from scipy.optimize import fmin_powell, minimize, minimize_scalar, basinhopping"""

'''
Connection to databases
'''

myclient   = pymongo.MongoClient("mongodb://localhost:27017/")
mydb       = myclient["pmt"]

ccol       = mydb["volcs"]     
pcol       = mydb["positions"]  
bcol       = mydb["volbs"] 
dfcol      = mydb["imgdfs"] 

    
'''

-----------------------------------------------------------------------------------------------
Calculator 1 :: FX Options
-----------------------------------------------------------------------------------------------

'''    

def pxo(curropt, currdt, extended=False, isusd=True, attribution=False):
    
    if attribution==True:
        ads = allDates(startday=4, startmonth=7, startyear=2018)
        prevdt = ads[ads.index(currdt)-1]
       
    
    """ Returns option holdings P&L change for given days
        in the HOLDINGS CURRENCY
        
           :: v_un = payoff (unsettled or mark-to-market)
           :: v_se = payoff (settled, only shown at expiry date, 'theoretical')
           :: p_un = cash payout (unsettled)
           :: p_se = cash payout (only shown at settle date) 
           
        p&l note : (v_se + p_se) + (p_un_chg_1d + v_un_chg_1d) = daily P&L   
           
        I . Discounting
        
        'curr', 'usym', discounting
        
        usym :: TRYUSD -> USDTRY tells you your black scholes calculation will be in 'TRY'
        curr :: tells you currency paid out or premium quotation currency.
        
        
        II. attribution
        
        V0 (previous mv), V1 (time shifted), V2 (spot + time shifted), just gives you and
        'alternative' V_un
           
    """
    
    NDs          = 'BRL,INR,IDR,PHP,KRW,TWD,MYR'.split(',')
    isND         = False
    if ((curropt['usym'][:3] in NDs) or (curropt['usym'][3:6] in NDs)):  
        isND = True
    marketparams = {}
    pltable      = {'v_un':[],  'v_se':[],  'p_un':[],  'p_se':[]}
    outputobject = {'v_un':0.0, 'v_se':0.0, 'p_un':0.0, 'p_se':0.0, 'emsg':''}
    outputobject['excludedLast'] = False
    if attribution:
        for keynew in ('v_0', 'v_1','v_2'):
            pltable[keynew] = []
            outputobject[keynew] = 0.0
    if extended: 
        outputobject['marketparams'] = marketparams
        outputobject['pltable']      = pltable
    
    longexpired  = (int(curropt['expiry']) <  int(currdt))
    atexpiry     = (int(curropt['expiry']) == int(currdt))
    notyet       = False
    if len(curropt['strips']) >= 1:
        if int(curropt['strips'][-1]['date']) > int(currdt):
            outputobject['emsg'] = 'future trades.'
            return outputobject
    if len(curropt['strips']) == 0:
        outputobject['emsg'] = 'not strips.'
        return outputobject
    
    """ 
    step 1. compute FV for a unit of FOR-currency 
    
    :: attribution=True, compute unitfv0 (original yesterday), unitfv1 (shifting only BS time), unitfv2 (shifting BS time and spot and fwd)
    
    """
    ticker        = curropt['usym'][3:6] + curropt['usym'][:3]
    currcp        = 1.0 if (curropt['cp'][0] in ('c','C')) else -1.0
    
    isRunning     = not(longexpired or atexpiry)
    
    if attribution:
        prevann  = ddf(prevdt, curropt['expiry'])/365.25
        prevspot = spotc(ticker, prevdt)
        if prevann > 0:
            prevfwd  = fwdc2(prevann, ticker, prevdt)
            prevol   = volac(ticker, curropt['stk'], prevann, prevdt)
        else:
            prevfwd  = spotc(ticker, prevdt)
            prevol   = 0.0
        
        dfusdo   = dfc(prevann, prevdt) 
        
    if isRunning:
        tenorann = ddf(currdt, curropt['expiry'])/365.25
        marketparams['fwd'] = currfwd  = fwdc2(tenorann, ticker, currdt)
        marketparams['vol'] = impvol   = volac(ticker, curropt['stk'], tenorann, currdt)
        marketparams['ufv'] = unitfv   = blk.bs(currcp, currfwd, curropt['stk'], impvol, tenorann, 1.0)
        marketparams['dfu'] = dfusd    = dfc(tenorann, currdt)
        if attribution:
            marketparams['ufv0'] = unitfv0  = blk.bs(currcp, prevfwd, curropt['stk'], prevol, prevann, 1.0)
            marketparams['ufv1'] = unitfv1  = blk.bs(currcp, prevfwd, curropt['stk'], prevol, tenorann, 1.0)
            marketparams['ufv2'] = unitfv2  = blk.bs(currcp, currfwd, curropt['stk'], prevol, tenorann, 1.0)
            
    elif (atexpiry):
        currspot      = spotc(ticker, currdt)
        dfusd         = 1.0
        if currcp > 0.0:
            unitfv   = max(currspot - curropt['stk'],0.0)
        else:
            unitfv   = max(curropt['stk'] - currspot,0.0)
            
        if attribution:
            unitfv0 = blk.bs(currcp, prevfwd, curropt['stk'], prevol, prevann, 1.0)
            if currcp > 0.0:
                unitfv1   = max(prevspot - curropt['stk'],0.0)
            else:
                unitfv1   = max(curropt['stk'] - prevspot,0.0)
            unitfv2 = unitfv
        marketparams['ufv']  = unitfv
        marketparams['spot'] = currspot
    else:
        unitfv = 0.0
        dfusd  = 1.0
        if attribution:
            if currcp > 0.0:
                unitfv0   = max(prevspot - curropt['stk'],0.0)
            else:
                unitfv0   = max(curropt['stk'] - prevspot,0.0)
            unitfv1 = unitfv2 = 0.0
            
    if attribution:
        marketparams['ufv0'], marketparams['ufv1'], marketparams['ufv2'] = unitfv0, unitfv1, unitfv2      
        
    """ 
    step 2. for each strip calculate p_se, p_un, v_se, v_un 
    
    """ 
    strips = sfstrip(curropt, currdt)
    for stripdx in range(len(strips)):
        
        strip = strips[stripdx]
        
        thisbuysell   = (-1.0 if (strip['type'] == 'Sell') else 1.0)
        
        '''
        
            step 1: calculate 'v'
            
        '''
        if ticker[3:6] != curropt['curr']:
            fxdomhld = spotc(ticker[3:6] + curropt['curr'], currdt)
            if attribution:
                fxdomhldo = spotc(ticker[3:6] + curropt['curr'], prevdt)
        else:
            fxdomhld, fxdomhldo = 1.0, 1.0
            
        marketparams['dfhld'] = dfhld  = dfusd
        dfhldo = dfusdo
        if isRunning and not(ticker[3:6] == 'USD'):
            dfhld  = spotc('USD' + ticker[3:6], currdt)/fwdc2(tenorann, 'USD' + ticker[3:6], currdt) * dfusd
            if attribution:
                dfhldo = spotc('USD' + ticker[3:6], prevdt)/fwdc2(prevann, 'USD' + ticker[3:6], prevdt) * dfusdo
                
            
        currpv        = unitfv * strip['qty'] * dfhld * fxdomhld
        
        if attribution:
            marketparams['upv0'] = unitfv0 * 1.0 * dfhldo * fxdomhldo
            
            currpv0   = unitfv0 * strip['qty'] * dfhldo * fxdomhldo
            currpv1   = unitfv1 * strip['qty'] * dfhldo * fxdomhldo
            currpv2   = unitfv2 * strip['qty'] * dfhld  * fxdomhld
        
        if isRunning:
            pltable['v_un'].append(thisbuysell * currpv)
            pltable['v_se'].append(0.0)
            if attribution:
                pltable['v_0'].append(thisbuysell * currpv0)
                pltable['v_1'].append(thisbuysell * currpv1)
                pltable['v_2'].append(thisbuysell * currpv2)
        elif (atexpiry):

            tradedate_at_expiry = (int(strip['date']) == int(curropt['expiry']) ) and (stripdx == 0)
            ''' 
                
                if tradedate_at_expiry = True (logic implemented 2019-05-03)
                    
                    .known as closed out trade, 
                    
                    :: non-deliverable : this carries cash so we need to net off the MV first
                    :: physical        : this carries NO cash so we do not have to net off MV
            
            '''
            
            if tradedate_at_expiry and not(isND):
                #print 'TRADE AT EXPIRY', curropt['id'], strip['dollarvalue']
                pltable['v_un'].append(0.0)
                pltable['v_se'].append(0.0)
                outputobject['excludedLast'] = True
                
            else:
                pltable['v_un'].append(0.0)
                pltable['v_se'].append(thisbuysell * currpv)
                if attribution:
                    pltable['v_0'].append(thisbuysell * currpv0)
                    pltable['v_1'].append(thisbuysell * currpv1)
                    pltable['v_2'].append(thisbuysell * currpv2)
            
        elif (longexpired):
            pltable['v_un'].append(0.0)
            pltable['v_se'].append(0.0)
        
        '''
        
            step 2: calculate 'p'
            
        '''
        
        if ((int(strip['settledt']) > int(currdt))):
            years_to_settle       = ddf(currdt, strip['settledt'])/365.25
            dfusd_settle          = dfc(years_to_settle, currdt)
            dfhld_settle          = dfusd_settle
            if not(curropt['curr'] == 'USD'):
                uxspot        = spotc('USD' + curropt['curr'], currdt)
                uxfwd         = fwdc2(years_to_settle, 'USD' + curropt['curr'], currdt)
                dfhld_settle  = uxspot/uxfwd * dfusd_settle
                #print uxspot, uxfwd
            pltable['p_un'].append(thisbuysell * -1.0 * strip['dollarvalue'] * dfhld_settle)
            pltable['p_se'].append(0.0)
        elif (int(strip['settledt']) == int(currdt)):
            pltable['p_un'].append(0.0)
            pltable['p_se'].append(thisbuysell * -1.0 * strip['dollarvalue'])
        else:
            pltable['p_un'].append(0.0)
            pltable['p_se'].append(0.0)
        
        for plkey in pltable.keys():
            sumpv = sum(pltable[plkey])
            
            if isusd and (curropt['curr']!='USD') and not((plkey in ['v_0', 'v_1'])):
                sumpv *= spotc(curropt['curr'] + 'USD', currdt)
            elif isusd and (curropt['curr']!='USD') and attribution and (plkey in ['v_0', 'v_1']):
                sumpv *= spotc(curropt['curr'] + 'USD', prevdt)
            outputobject[plkey] = sumpv
            
    return outputobject
    
'''

-----------------------------------------------------------------------------------------------
Calculator 2 :: FX Forwards
-----------------------------------------------------------------------------------------------

'''    
def pxf(currfwd, currdt, extended=False, isusd=True, attribution=None):    

    """ Returns FX FORWARD holdings P&L change for given days
        in the HOLDINGS CURRENCY
        
           :: v_un = payoff (unsettled or mark-to-market)
           :: v_se = payoff (settled, only shown at expiry date, 'theoretical')
           :: p_un = cash payout (unsettled)
           :: p_se = cash payout (only shown at settle date) 
           
        p&l note : (v_se + p_se) + (p_un_chg_1d + v_un_chg_1d) = daily P&L   
           
    """
    
    NDs          = 'BRL,INR,IDR,PHP,KRW,TWD,MYR'.split(',')
    isND         = False
    if ((currfwd['usym'][:3] in NDs) or (currfwd['usym'][3:6] in NDs)):  
        isND = True
    marketparams = {}
    pltable      = {'v_un':[],  'v_se':[],  'p_un':[],  'p_se':[]}
    outputobject = {'v_un':0.0, 'v_se':0.0, 'p_un':0.0, 'p_se':0.0, 'emsg':''}
    if extended: 
        outputobject['marketparams'] = marketparams
        outputobject['pltable']      = pltable
    
    longexpired  = (int(currfwd['expiry']) <  int(currdt))
    atexpiry     = (int(currfwd['expiry']) == int(currdt))
    
    """ step 1. holding level market data """
    ticker        = currfwd['usym'][3:6] + currfwd['usym'][:3]
    
    isRunning     = not(longexpired or atexpiry)
    
    tenorann      = ddf(currdt,currfwd['expiry'])/365.25 
    
    mktfwd        = 0.0
    dfusd         = 1.0 
    
    # need to understand how you are using the 'currfwd'
    
    if isRunning:
        mktfwd        = fwdc2(tenorann, ticker, currdt)
        dfusd         = dfc(tenorann, currdt)
    else:
        mktfwd        = spotc(ticker, str(currfwd['expiry']))
        dfusd         = 1.0
        
    marketparams['mktfwd'] = mktfwd # e.g. GBPBRL
    '''
    if isRunning:
        tenorann = ddf(currdt, currfwd['expiry'])/365.25
        marketparams['fwd'] = currfwd  = fwdc2(tenorann, ticker, currdt)
        marketparams['vol'] = impvol   = volac(ticker, currfwd['stk'], tenorann, currdt)
        marketparams['ufv'] = unitfv   = blk.bs(currcp, currfwd, currfwd['stk'], impvol, tenorann, 1.0)
        marketparams['dfu'] = dfusd    = dfc(tenorann, currdt)
        
    elif (atexpiry):
        currspot      = spotc(ticker, currdt)
        dfusd         = 1.0
        if currcp > 0.0:
            unitfv   = max(currspot - currfwd['stk'],0.0)
        else:
            unitfv   = max(currfwd['stk'] - currspot,0.0)
        
        marketparams['spot'] = currspot
    else:
        unitfv = 0.0
        dfusd  = 1.0
    '''    
    """ step 2. for each strip calculate p_se, p_un, v_se, v_un """ 
    
    strips = sfstrip(currfwd, currdt)
    for stripdx in range(len(strips)):
        
        strip = strips[stripdx]
        
        thisbuysell   = (-1.0 if (strip['type'] == 'Sell') else 1.0)
        
        '''
        
            step 1: calculate 'v'
            
        '''
        if ticker[3:6] != currfwd['curr']:
            fxdomhld = spotc(ticker[3:6] + currfwd['curr'], currdt)
        else:
            fxdomhld = 1.0
            
        marketparams['dfhld'] = dfhld  = dfusd
        
        if isRunning and not(ticker[3:6] == 'USD'):
            dfhld  = spotc('USD' + ticker[3:6], currdt)/fwdc2(tenorann, 'USD' + ticker[3:6], currdt) * dfusd
            
        currpv        = strip['qty'] * dfhld * (mktfwd - float(strip['cost']))        
        
        if isRunning:
        
            pltable['v_un'].append(thisbuysell * currpv)
            pltable['v_se'].append(0.0)
            
        elif (atexpiry):

            tradedate_at_expiry = int(strip['date']) == int(currfwd['expiry']) and (stripdx == 0)
            ''' 
                
                if tradedate_at_expiry = True (logic implemented 2019-05-03)
                    
                    .known as closed out trade, 
                    
                    :: non-deliverable : this carries cash so we need to net off the MV first
                    :: physical        : this carries NO cash so we do not have to net off MV
            
            '''
            if tradedate_at_expiry and not(isND):
                #print currfwd['id'], strip['dollarvalue']
                pltable['v_un'].append(0.0)
                pltable['v_se'].append(0.0) #thisbuysell * currpv)
            else:
                pltable['v_un'].append(0.0)
                pltable['v_se'].append(thisbuysell * currpv)
            
        elif (longexpired):
            pltable['v_un'].append(0.0)
            pltable['v_se'].append(0.0)
        
        '''
        
            step 2: calculate 'p'
            
            
        if ((int(strip['settledt']) > int(currdt))):
            years_to_settle       = ddf(currdt, strip['settledt'])/365.25
            dfusd_settle          = dfc(years_to_settle, currdt)
            dfhld_settle          = dfusd_settle
            if not(currfwd['curr'] == 'USD'):
                dfhld_settle  = spotc('USD' + currfwd['curr'], currdt)/fwdc2(years_to_settle, 'USD' + currfwd['curr'], currdt) * dfusd_settle
            pltable['p_un'].append(thisbuysell * -1.0 * strip['dollarvalue'] * dfhld_settle)
            pltable['p_se'].append(0.0)
        elif (int(strip['settledt']) == int(currdt)):
            pltable['p_un'].append(0.0)
            pltable['p_se'].append(thisbuysell * -1.0 * strip['dollarvalue'])
        else:
            pltable['p_un'].append(0.0)
            pltable['p_se'].append(0.0)
        
        '''  
        for plkey in pltable.keys():
            sumpv = sum(pltable[plkey])
            if isusd and (currfwd['curr']!='USD'):
                sumpv *= spotc(currfwd['curr'] + 'USD', currdt)
            outputobject[plkey] = sumpv
      
    return outputobject


'''

-----------------------------------------------------------------------------------------------
Calculator 3 :: Variance Swap and Volatility Swap 
-----------------------------------------------------------------------------------------------

'''    

def pxv(vobj, cdt, extended=False, isusd=True, volswapmethod='cl', varswapmethod=True, attribution=True):
    if attribution==True:
        ads = allDates(startday=4, startmonth=7, startyear=2018)
        prevdt = ads[ads.index(cdt)-1]
    pltable = {}
    output  = {'emsg':'','marketparams':{},'pltable':pltable}
    if attribution:
        plkeys  = ('p_se', 'p_un', 'v_se', 'v_un', 'v_th')
    else:
        plkeys  = ('p_se', 'p_un', 'v_se', 'v_un')
    for key in plkeys:
        output[key] = 0.0
        pltable[key] = []
        
    ticker = vobj['usym'].split('.')[0]
    
    longexpired  = (int(vobj['expiry']) <  int(cdt))
    atexpiry     = (int(vobj['expiry']) == int(cdt))
    
    if len(ticker) != 6: 
        output['emsg'] += 'wrongticker|'
        return output
    elif vobj['firstdate'] > cdt:
        output['emsg'] += 'firsttrade in future|'
        return output
    else:
        if ticker[:3] == vobj['smccy']:
            ticker = ticker[3:6] + ticker[0:3]
    
    '''
    first define the 'final vol'
    '''
    finalvol = 0.0
    matann    = ddf(cdt, vobj['lastdate'])/365.25
    fwdann    = ddf(cdt, vobj['firstdate'])/365.25
    
    isreal    = True  if int(cdt) > int(vobj['firstdate']) else False 
    isfwd     = True  if fwdann > 0.0 else False 
    isvolswp  = False if vobj['type'] == 'VSWP' else True
    
    ''' 
    implied sigma
    '''
    impvol, impvolpp    = 0.0, 0.0
    swaptenor = matann if isreal else matann - fwdann 
    if (swaptenor > 0):
        impvol    = varhelper(ticker, swaptenor, cdt, fwdann=(0 if isreal else fwdann), isvolswap=isvolswp, volswapmethod=volswapmethod, varswapmethod=varswapmethod)
        if attribution: # attribution 1 
            # impvolpp = implied vol with previous parameter
            impvolpp    = varhelper(ticker, swaptenor, prevdt, fwdann=(0 if isreal else fwdann), isvolswap=isvolswp, volswapmethod=volswapmethod, varswapmethod=varswapmethod)

            if not(isvolswp):
                impvolpp    = impvolpp  ** 0.5 
        if not(isvolswp):
            impvol = impvol ** 0.5
        
    ''' 
    realised sigma
    '''
    realvol   = 0.0
    isRunning = not(longexpired) and not(atexpiry)
    if isreal and (isRunning):
        realvolobj = realisedvol2(ticker, str(vobj['firstdate']), (str(cdt)), extended=True)
        realvol    = realvolobj['vol']
    elif isreal and not(isRunning):
        realvolobj = realisedvol2(ticker, str(vobj['firstdate']), str(vobj['lastdate']), extended=True)
        realvol    = realvolobj['vol']
    else: 
        None
        
    ''' 
    final sigma 
    '''
    if True:
        futuredays, daysadj = 0, 1.
        for schdt in vobj['fullschedule']:
            if int(schdt) > int(cdt):
                futuredays += 1
        if futuredays > 1:
            daysadj  = ( 252 * swaptenor ) / (futuredays - 1)
       
        if vobj['type'] == 'vSWP' and (realvol >= float(vobj['stk'])*.01):
            finalvol = ((realvol ** 1 * -1. * fwdann) + (impvol ** 1 * swaptenor * daysadj))/(swaptenor - (fwdann if isreal else 0.0))
            if attribution:
                finalvolpp = ((realvol ** 1 * -1. * fwdann) + (impvolpp ** 1 * swaptenor * daysadj))/(swaptenor - (fwdann if isreal else 0.0))
        else:
            finalvar = ((realvol ** 2 * -1. * fwdann) + (impvol ** 2 * swaptenor * daysadj))/(swaptenor - (fwdann if isreal else 0.0))
            finalvol = finalvar ** .5
            if attribution:
                finalvarpp = ((realvol ** 2 * -1. * fwdann) + (impvolpp ** 2 * swaptenor * daysadj))/(swaptenor - (fwdann if isreal else 0.0))
                finalvolpp = finalvarpp ** .5
        
    curstage = ('season' if isreal else '') + ('forward' if isfwd else '') + ('inception' if (not(isfwd) and not(isreal)) else '') + ' ' + ('varSwap' if (vobj['type'] == 'VSWP') else 'volSwap')
    marketparams = {'parSigma':finalvol, 'impSigma':impvol,'realSigma':realvol, 'curstage':curstage, 'daysadj':daysadj}
    if attribution:
        marketparams['pp_impSigma'] = impvolpp
        marketparams['pp_parSigma'] = finalvolpp
        #print impvolpp, impvol, finalvolpp, finalvol, realvol
    
    output['marketparams'] = marketparams 
    
    
    '''
    Price the contract (per strip)
    '''
    strips = sfstrip(vobj, cdt)
    marketparams['dfhld'] = dfhld = 1.0
    marketparams['dfhldpp']   = dfhldpp  = 1.0
    dfusd = dfc(matann, cdt)
    dfusdpp = dfc(matann, prevdt)
    if isRunning and not(vobj['smccy'] == 'USD'):
        marketparams['dfhld']   = dfhld  = spotc('USD' + vobj['smccy'] , cdt)/fwdc2(matann, 'USD' + vobj['smccy'], cdt) * dfusd
        marketparams['dfhldpp']   = dfhldpp  = spotc('USD' + vobj['smccy'] , prevdt)/fwdc2(matann, 'USD' + vobj['smccy'], prevdt) * dfusdpp
    elif (atexpiry):
        dfhld = dfusd = 1.0
        dfhldpp = dfusdpp = 1.0
        
    for stripdx in range(len(strips)):
        
        strip         = strips[stripdx]
        
        thisbuysell   = (-1.0 if (strip['type'] == 'Sell') else 1.0)
        
        ###step 1: calculate 'v'
        
        if vobj['type'] == 'vSWP':
            currpv        = vobj['vegaNotional']  * dfhld * (finalvol * 100. - float(vobj['stk'])) 
            if attribution:
                pppv        = vobj['vegaNotional']  * dfhldpp * (finalvolpp * 100. - float(vobj['stk'])) 
        else:
            currpv        = vobj['varianceUnits'] * dfhld * ((finalvol * 100.)**2 - (float(vobj['stk']))**2) 
            if attribution:
                pppv        = vobj['varianceUnits'] * dfhldpp * ((finalvolpp * 100.)**2 - (float(vobj['stk']))**2) 
        
        if not(atexpiry) and not(longexpired):
            pltable['v_un'].append(thisbuysell * currpv)
            pltable['v_se'].append(0.0)
            if attribution:
                pltable['v_th'].append(thisbuysell * pppv)
        elif (atexpiry):
            pltable['v_un'].append(0.0)
            pltable['v_se'].append(thisbuysell * currpv)
            if attribution:
                pltable['v_th'].append(thisbuysell * pppv)
        elif (longexpired):
            pltable['v_un'].append(0.0)
            pltable['v_se'].append(0.0)
            
        ###    step 2: calculate 'p'
        
        if ((int(strip['settledt']) > int(cdt))):
            years_to_settle       = ddf(cdt, strip['settledt'])/365.25
            dfusd_settle          = dfc(years_to_settle, cdt)
            dfhld_settle          = dfusd_settle
            if not(vobj['smccy'] == 'USD'):
                uxspot        = spotc('USD' + vobj['smccy'], cdt)
                uxfwd         = fwdc2(years_to_settle, 'USD' + vobj['smccy'], cdt)
                dfhld_settle  = uxspot/uxfwd * dfusd_settle
                #print uxspot, uxfwd
            pltable['p_un'].append(thisbuysell * -1.0 * strip['dollarvalue'] * dfhld_settle)
            pltable['p_se'].append(0.0)
        elif (int(strip['settledt']) == int(cdt)):
            pltable['p_un'].append(0.0)
            pltable['p_se'].append(thisbuysell * -1.0 * strip['dollarvalue'])
        else:
            pltable['p_un'].append(0.0)
            pltable['p_se'].append(0.0)
        
        for plkey in pltable.keys():
            sumpv = sum(pltable[plkey])
            
            if isusd and (vobj['smccy'] !='USD'):
                if plkey == 'v_th':
                    spotnow = spotc(vobj['smccy'] + 'USD', prevdt)
                else:
                    spotnow = spotc(vobj['smccy'] + 'USD', cdt)
                    
                sumpv *= spotnow
            output[plkey] = sumpv
 
    return output
 