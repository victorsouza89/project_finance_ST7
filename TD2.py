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


def mu_estimate(date):
    perf_list=pd.read_csv('performance.csv',sep=';')
    sedol_list=perf_list.keys()[1:]
    dates=perf_list['Dates']
    j=0
    for i in range(len(dates)):
        if str(dates[i]+' 00:00:00')==str(date):
            j=i
    if j>=5000:
        return [np.mean([perf_list[sedol][i] for i in range(j-600,j) ]) for sedol in   sedol_list  ]
    else:
        return [0 for _ in range(len(sedol_list))]



def sigma_estimate(date):
    return main.get_cov(date)

#a=sigma_estimate('2021-10-31')
#np.savetxt('test.csv', a, delimiter=';') 

def get_weight(date,lamb=1):
    mu=mu_estimate(date)
    sigma=sigma_estimate(date)
    n=len(mu)
    if sigma[0][0]==0:
        return [0 for _ in range(n)],0,0
    w = cp.Variable(n)
    gamma = lamb
    ret = mu@w 
    risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Maximize(ret - gamma*risk),[cp.sum(w) == 1, 
                w >= 0])
    prob.solve()
    
    return w.value,risk.value,ret.value 

#print(get_weight('2021-10-31'))

def get_all_weights(df2):
    all_dates=df2['date']
    liste=[]
    for date in all_dates:
        print(date)
        try:
            w,risk,perf=get_weight(date)
            liste.append([str(date),str(perf),str(risk)]+[str(x) for x in w])

        except:
            print('fail')
            
    return liste

liste=get_all_weights(df2)
np.savetxt('weight_opti.csv', liste, delimiter=';') 







