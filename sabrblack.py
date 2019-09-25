#sabrblack.py

from black import bs, bsgreek
from sabrme2 import smile

# extension to the sabr and black family 

def sabrgreek(stk, t, sabrparams, type='rho'):
    
    bump = {'rho':[0.0, 0.01, 0.0], 'nu':[0.0, 0.0, 0.01], 'alpha':[0.01, 0.0, 0.0]}
    
    cursmile = smile(1., t, 1.)
    
    cursmile.params = sabrparams
    
    vol0 = cursmile.modelvols([stk,])[0]
    
    bs0  = bs(1., 1., stk, vol0, t, 1.) 
    
    '''
    
    bump need to set ATMF the same..
    
    '''
    
    if type != 'gamma':
        # find out atmv 
        atmv = cursmile.modelvols([1.,])[0]
        
        # new smile
        smile2 = smile(1., t, 1.)
        
        # pin atmv, and old params 
        smile2.fit3([1.,], [atmv + bump[type][0],], {'atmv':atmv + bump[type][0], 'rho':cursmile.params['rho'] + bump[type][1], 'nu':cursmile.params['nu'] + bump[type][2]})
        
        # new vol 
        vol1 = smile2.modelvols([stk,])[0]
        
        # new price 
        bs1  = bs(1., 1., stk, vol1, t, 1.) 
        
        #print vol1, vol0
        
        return bs1 - bs0
    
    else:
        smileup = smile(1.01, t, 1.0)
        smiledn = smile(0.99, t, 1.0)
        
        smileup.params = sabrparams
        smiledn.params = sabrparams
        
        volu = smileup.modelvols([stk,])[0]
        vold = smiledn.modelvols([stk,])[0]
        
        return bs(1., 1.01, stk, volu, t, 1.) + bs(1., 0.99, stk, vold, t, 1.)  - 2* bs0 #/(.01 * fwd)

class bso:

    def __init__(self, cp, fwd, stk, vol, t, df):
    
        self.cp  = cp
        self.fwd = fwd
        self.stk = stk
        self.t   = t
        self.vol = vol
        self.df  = df
        self.pv    = bs(self.cp, self.fwd, self.stk, self.vol, self.t, self.df)
        self.delta = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='delta')
        self.theta = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='theta')
        self.vega  = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='vega')
        self.volgamma = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='volgamma')
        self.gamma = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='gamma')
        
    def calc(self):
        
        self.pv    = bs(self.cp, self.fwd, self.stk, self.vol, self.t, self.df)
        self.delta = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='delta')
        self.theta = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='theta')
        self.vega  = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='vega')
        self.volgamma = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='volgamma')
        self.gamma = bsgreek(self.cp, self.fwd, self.stk, self.vol, self.t, self.df, type='gamma')
        
        return 'calcdone'
        
    def calcsabr(self, sabrparams):
        self.srho = sabrgreek(self.stk, self.t, sabrparams, type='rho')
        self.snu  = sabrgreek(self.stk, self.t, sabrparams, type='nu')
        self.svega = sabrgreek(self.stk, self.t, sabrparams, type='alpha')
        self.sgamma = sabrgreek(self.stk, self.t, sabrparams, type='gamma')
        
        return 'calcSabrdone'
# new! portfolio class, a series of option OBJECTS 
        
class bsp:
        
        '''
        
            input: {'notional':500, 'option':bso1}
            
        '''
        
        def __init__(self, positions):
            self.positions       = positions
            self.xpositions      = [] 
            self.spositions      = []
        
        def calc(self):
        
            self.pv    = 0.0
            self.delta = 0.0
            self.theta = 0.0
            self.vega  = 0.0
            self.volgamma = 0.0
            self.gamma = 0.0
            self.xpositions      = [] 
            
            for pos in self.positions:
                pos['option'].calc()
                thispv    = pos['option'].pv * pos['notional']
                thisdelta = pos['option'].delta * pos['notional']
                thistheta = pos['option'].theta * pos['notional']
                thisvega  = pos['option'].vega * pos['notional']
                thisvolgamma = pos['option'].volgamma * pos['notional']
                thisgamma  = pos['option'].gamma * pos['notional']
                
                self.xpositions.append({'optStk':pos['option'].stk, 'optT':pos['option'].t, 'optVol':pos['option'].vol, 'notional':pos['notional'], 
                'delta':thisdelta, 'gamma':thisgamma, 'theta': thistheta, 'vega':thisvega, 'volgamma': thisvolgamma, 'pv': thispv})
                
                self.pv    += thispv
                self.delta += thisdelta
                self.theta += thistheta
                self.vega  += thisvega
                self.volgamma += thisvolgamma
                self.gamma += thisgamma
                
        def calcsabr(self):
        
            self.srho  = 0.0
            self.svega = 0.0
            self.snu   = 0.0
            self.sgamma   = 0.0
            self.spositions      = []
            
            for pos in self.positions:
            
                pos['option'].calc()
                pos['option'].calcsabr(pos['params'])
                
                thisrho    = pos['option'].srho * pos['notional']
                thisnu     = pos['option'].snu * pos['notional']
                thisalpha  = pos['option'].svega * pos['notional']
                thisgamma  = pos['option'].sgamma * pos['notional']
                
                self.spositions.append({'optStk':pos['option'].stk, 'optT':pos['option'].t, 'optVol':pos['option'].vol, 'notional':pos['notional'], 
                'svega':thisalpha, 'srho':thisrho, 'snu': thisnu, 'sgamma': thisgamma})
                
                self.snu     += thisnu
                self.srho    += thisrho
                self.svega   += thisalpha
                self.sgamma   += thisgamma
            