from pathlib import Path
import csv, math, mpmath as mp
mp.mp.dps=100
dest=Path(__file__).parent
dest.mkdir(exist_ok=True)
rows=[]
for phi in [.3,2.,100.,1e8,1e12,1e15]:
    y=1+.5/math.sqrt(phi)
    p=mp.mpf(phi); yy=mp.mpf(y)
    value=p*mp.log(p)-mp.loggamma(p)+(p-1)*mp.log(yy)-p*yy
    dphi=mp.log(p)+1-mp.digamma(p)+mp.log(yy)-yy
    dphi2=1/p-mp.polygamma(1,p)
    rows.append([phi,y,float(value),float(dphi),float(dphi2)])
with (dest/'ratio-gamma-precision.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['phi','y','expected','dphi','dphi2']);w.writerows(rows)
rows=[]
for phi in [.3,2.,100.,1e8,1e12,1e15]:
    n=40; y=round(20+.5*math.sqrt(n*.25*(n+phi)/(1+phi)))
    p=mp.mpf(phi); a=p/2
    value=mp.loggamma(n+1)-mp.loggamma(y+1)-mp.loggamma(n-y+1)+mp.loggamma(a+y)+mp.loggamma(a+n-y)-mp.loggamma(p+n)-2*mp.loggamma(a)+mp.loggamma(p)
    dphi=(mp.digamma(a+y)+mp.digamma(a+n-y))/2-mp.digamma(p+n)-mp.digamma(a)+mp.digamma(p)
    dphi2=(mp.polygamma(1,a+y)+mp.polygamma(1,a+n-y))/4-mp.polygamma(1,p+n)-mp.polygamma(1,a)/2+mp.polygamma(1,p)
    rows.append([phi,y,n,float(value),float(dphi),float(dphi2)])
with (dest/'ratio-beta-binomial-precision.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['phi','y','n','expected','dphi','dphi2']);w.writerows(rows)
