import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
import main
import pickle

#import main.py as main

"""ex1"""
'a'
volatility = 0.01*np.array([[14.3,17.4,21.2,4.3,4,8.4,0.5]])
mu = np.array([[6,7,9.5,1.5,1.3,3.2,0]])
rho =  np.array([[1,0.82,0.78,0.1,0,0.5,0],[0.82,1,0.85,0.12,0.08,0.63,0],[0.78,0.85,1,0.05,0.03,0.71,0],[0.1,0.12,0.05,1,0.65,0.2,0],[0,0.08,0.03,0.65,1,0.23,0],[0.5,0.63,0.71,0.2,0.23,1,0],[0,0,0,0,0,0,1]])
path = "DataProjets.xlsx"
df2 = pd.read_excel(path, sheet_name="MarketCaps")
df2 = df2.rename(columns={"Unnamed: 0": "date"})
base=pd.read_csv('indicators.csv',sep=';')
opti_weights=pd.read_csv('weights_opti.csv',sep=';')
def get_delta(rho,volatility):
    n=len(volatility[0])
    delta=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            delta[i,j]=rho[i,j]*volatility[0,i]*volatility[0,j]
    return delta

def weight(volatily=volatility,mu=mu,rho=rho,lambd=0.5):
    delta=get_delta(rho,volatility)
    return 1/(2*lambd)*np.dot(mu,inv(delta))

def get_volatility(weight,rho):
    v=0
    for i in range(len(weight)):
        for j in range(len(weight)):
            v+=weight[0,i]*weight[0,j]*rho[i,j]
    return v

def get_return(weight,mu):
    v=0
    return (mu@weight.T)[0,0]

w=weight()
print("weight")
print(w)  
print("volatility")
print(get_volatility(w,rho))
print("return")
print(get_return(w,mu))
'c'
delta=[]
def weight_cp(volatily=volatility,mu=mu,delta=delta,gamma=0.5):
    n=len(volatility[0])
    delta=get_delta(rho,volatility)
    w = cp.Variable(n)
    ret = mu@w 
    risk = cp.quad_form(w, delta)
    prob = cp.Problem(cp.Maximize(ret - gamma*risk))
    prob.solve()
    return w.value

print(weight_cp())


"""ex3"""

retur=[]
vol=[]
for lam in np.linspace(150,1000,10000):
    w=weight(lambd=lam)
    retur.append(get_return(w,mu))
    vol.append(get_volatility(w,rho))
    '''
plt.plot(vol,retur)
plt.show()
'''
"""ex4"""
def weight_constraint_cp(lamb,rho,volatility,mu,delta):
    n=len(volatility[0])
    delta=get_delta(rho,volatility)
    w = cp.Variable(n)
    gamma = lamb
    ret = mu@w 
    risk = cp.quad_form(w, delta)
    prob = cp.Problem(cp.Maximize(ret - gamma*risk),[cp.sum(w) == 1, 
                w >= 0])
    prob.solve()
    
    return w.value,risk.value,ret.value 

retur=[]
vol=[]
for lam in np.linspace(0.001,5000,500):
    w,v,r=weight_constraint_cp(lam,rho,volatility,mu,delta)
    retur.append(r)
    vol.append(v)
    '''
plt.plot(vol,retur)
plt.show()
'''

"""Projet"""

perf_list=pd.read_csv('performance.csv',sep=';')
def mu_estimate(date,perf_list=perf_list):

    sedol_list=perf_list.keys()[1:]
    dates=perf_list['Dates']
    j=0
    for i in range(len(dates)):
        if str(dates[i]+' 00:00:00')==str(date):
            j=i
    if j>=300:
        return [np.mean([perf_list[sedol][i] for i in range(j-300,j) ]) for sedol in   sedol_list  ]
    else:
        return [0 for _ in range(len(sedol_list))]



def sigma_estimate(date):
    return main.get_cov(date)

#a=sigma_estimate('2021-10-31')
#np.savetxt('test.csv', a, delimiter=';') 

def get_weight(date,volatility):
    mu=mu_estimate(date)
    n=len(mu)
    sigma=sigma_estimate(date)
    if sigma[0][0]==0:
        return [0 for _ in range(n)],0,0
    sigma=sigma+0.000001*np.identity(n)
    w = cp.Variable(n)
    ret = mu@w 
    risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Maximize(ret),[cp.sum(w) == 1, 
                w >= 0,risk<=volatility  ])
    prob.solve()
    
    return w.value,risk.value,ret.value 

#print(get_weight('2021-10-31'))

def get_all_weights(df2,base=base):
    all_dates=df2['date']
    liste=[]
    for date in all_dates:
        print(date)
        try:
            volatility=base[str(date)[0:10]][1]
            w,risk,perf=get_weight(date,volatility)
            liste.append([str(date)[0:10],str(perf),str(risk)]+[str(x) for x in w])
            print(risk,perf)
        except:
            print('fail')
    return liste

'''
liste=get_all_weights(df2)
import csv
with open('output.csv','w',newline="") as result_file:
    wr = csv.writer(result_file,delimiter=";")
    wr.writerows(liste)
'''

def performance_sector_opti(sector):
    dates=df2["date"]
    columns = df2.columns[1:]
    r={}
    for j,date in enumerate(dates):
        try:
            companies_sector=main.sector_sorting(date)[sector]
            mu=main.mu_estimate_sector(date,companies_sector,lg=300)
            rt=0
            company_weight=[]
            for c in companies_sector:
                company_weight.append(d[c][j])
            for (i,c) in enumerate(companies_sector):
               
                w_t_i = company_weight[j]
                r_t_i = mu[i]
                rt+=w_t_i*r_t_i
            print(rt)
            r[date]=[rt]

        except:
            print('fail')
            r[date]=[0]
        
    return r

    


def risk_sector_opti(sector):
    dates=df2["date"]
    columns = df2.columns[1:]
    risque={}
    for j,date in enumerate(dates):
        try:
            companies_sector=main.sector_sorting(date)[sector]
            cov=cov_estimate_sector(date,companies_sector,lg=300)
            risquet=0
            if cov[0][0]==0:
                risquet=0
                print(0)
            else:
                for (i1,c1) in enumerate(companies_sector):
                    w1=NormalizeData(df2[c1])
                    for (i2,c2) in enumerate(companies_sector):
                        w2=NormalizeData(df2[c2])

                        risquet+=cov[i1,i2]*w1[j]*w2[j]
                print(risquet)
                risque[date]=[risquet]
        except:
            risque[date]=[0]
            print('fail')
        
    return risque


