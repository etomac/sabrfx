import datetime as dt
from math import log

def calcdate(datenow, tenor):
    
    nbperiod = float(tenor[:-1])
    period   = tenor[-1]
    
    conv = {'D':1, 'W':7, 'M':30.5, 'Y':365}

    return (datenow + dt.timedelta(nbperiod*conv[period]))
    
    
def xaxis(s, vol, t, sd=3., nb=41):

    down = s * ( 1 - sd * vol * (t ** 0.5))
    up   = s * ( 1 + sd * vol * (t ** 0.5))
    

    #base = 10 ** round(log(s,10))
    #lmin = round(down * 10./base)/10.*base
    #lmax = round(up * 10./base)/10.*base
    lmin, lmax = down, up
    return [lmin + i * (lmax - lmin)/nb for i in range(nb)]
    
    
def main():
    print calcdate(dt.datetime.today(), "1Y")
    print xaxis(100, 0.2, 2)
    
    
if __name__=="__main__": 
    main()