import numpy as np
import datetime as dt
from scipy.interpolate import interp1d, CubicSpline as spline, InterpolatedUnivariateSpline as spline2

def interpd(datenow, datetext, valy, dateintp, schextp="flat", schintp="linear"):

    valx0    = dt.datetime.strptime(datenow, '%Y%m%d')
    valx     = np.array([(dt.datetime.strptime(d, '%Y%m%d') - valx0).days for d in datetext]) 
    
    valxintp = np.array([(dt.datetime.strptime(dateintp, '%Y%m%d') - valx0).days])

    minmax = (valy[np.argmin(valx)], valy[np.argmax(valx)])

    try:
        if schextp == 'flat' and valxintp.tolist()[0] <= np.min(valx):
            return minmax[0]
        elif schextp == 'flat' and valxintp.tolist()[0] >= np.max(valx):
            return minmax[1]
        else: 
            f = interp1d(valx, valy, fill_value='extrapolate', kind=schintp)
            return f(valxintp).tolist()[0]

    except ValueError:
        return 'ValueError, try increasing observations'
        
def interp(valx, valy, xintp, schextp="flat", schintp="linear"):

    valx     = np.array(valx) 
    valxintp = np.array([xintp])

    minmax = (valy[np.argmin(valx)], valy[np.argmax(valx)])

    try:
        if schextp == 'flat' and valxintp.tolist()[0] <= np.min(valx):
            return minmax[0]
        elif schextp == 'flat' and valxintp.tolist()[0] >= np.max(valx):
            return minmax[1]
        else: 
            f = interp1d(valx, valy, fill_value='extrapolate', kind=schintp)
            return f(valxintp).tolist()[0]

    except ValueError:
        return 'ValueError, try increasing observations'

if __name__ == "__main__":
    
    # exercise 1: testing function 

    datetext = ['20181201','20190301','20180601','20200301','20210301']
    valy     = [0.99, 0.98, 0.995, 0.96, 0.95]
    datenow  = '20180501'
    dateintp = '20290502'
    schextp  = 'flat'
    schintp  = 'cubic'

    print interpd(datenow, datetext, valy, dateintp, schextp, schintp)
    print interp([1,2,3,4,5],valy,4.745)
    
    # exercise 2: cubic spline extrapolation
    
    vvx = [0.513, 0.523, 0.537, 0.55, 0.556]
    vvy = [9.32, 8.29, 7.33, 7.07, 6.84]
    
    vix = np.linspace(0.4, 0.8, 100)
    # viy = interp(vvx, vvy, vix, schextp='n', schintp='cubic')
    
    viy = spline2(vvx, vvy, k=3)
    viy = viy(vix)
    for idx in range(len(vix)):
        print vix[idx], viy[idx]