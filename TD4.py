import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cvxpy as cp
import pandas as pd
from copy import deepcopy
import main
from math import sqrt
import pickle 
import matplotlib.pyplot as plt

path = "DataProjets.xlsx"
df2 = pd.read_excel(path, sheet_name="MarketCaps")
df2 = df2.rename(columns={"Unnamed: 0": "date"})
all_dates = df2['date']
perf=pd.read_csv('performance.csv',sep=';')
volatility = np.array([[14.3, 17.4, 21.2, 4.3, 4, 8.4, 0.5]])
mu = np.array([[6, 7, 9.5, 1.5, 1.3, 3.2, 0]])
rho = np.array(
    [
        [1, 0.82, 0.78, 0.1, 0, 0.5, 0],
        [0.82, 1, 0.85, 0.12, 0.08, 0.63, 0],
        [0.78, 0.85, 1, 0.05, 0.03, 0.71, 0],
        [0.1, 0.12, 0.05, 1, 0.65, 0.2, 0],
        [0, 0.08, 0.03, 0.65, 1, 0.23, 0],
        [0.5, 0.63, 0.71, 0.2, 0.23, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
)
kappa = 0.2


"Exercice 1"

def market_cap(date,n,ent=150):
    path = "marketCaps.csv"
    df = pd.read_csv(path,index_col=0,sep=';')
    print(date)

    data_cap = df.loc[str(date)]

    daily_data_cap = np.zeros(n)
    res = np.zeros((n,n))
    for i in range(n):
        daily_data_cap[i] = data_cap[i]
    sorted_data = np.sort(daily_data_cap)
    boundary = sorted_data[-ent]

    for i in range(n):
        if daily_data_cap[i] < boundary:
            res[i,i] = 1
    return res


def weight_cp_mean(sigma, mu, kappa, date, ent):
    date=date[0:10]

    n = len(sigma[0])
    m = len(sigma)

    omega = np.zeros((m, m))
    for i in range(m):
        omega[i][i] = sigma[i][i]


    w = cp.Variable(n)
    ret = np.array(mu).T @ w

    omegaEigenvalues, omegaEigenvectors = np.linalg.eig(omega)
    omegaD = np.diag(omegaEigenvalues)

    ret2 = cp.norm(np.sqrt(omegaD) @ omegaEigenvectors @ w)


    vol_mean = cp.quad_form(w,sigma)

    better_cap = market_cap(date,n,ent)
    constraints_cap = w @ better_cap

    constraints = [cp.sum(w) == 1, w >= 0, vol_mean <= 0.04/365,sum(constraints_cap) == 0]

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2), constraints)
    prob.solve()

    return w.value, w, prob, ret.value, ret2, vol_mean.value



def portfolio_contraint(date,ent):
    sigma_mean = main.get_cov(date)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return 0,0,0,0,0
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    w_rob,w,prob,ret,ret2,vol_mean = weight_cp_mean(sigma_mean,mu_mean,0.25,date,ent)
    return w_rob,w,prob,ret,ret2, vol_mean



"""Partie 2 : lambda equivalent a detendre"""

def weight_cp_mean_lambda(w_opti, sigma, mu, kappa, date, lambd_opti, ent=150):
    date=date[0:10]

    n = len(sigma[0])
    m = len(sigma)
    

    omega = np.zeros((m, m))
    for i in range(m):
        omega[i][i] = sigma[i][i]


    try:
        lambd = (np.array(w_opti).T @ sigma @ np.array(w_opti))**(-1) * (np.array(mu).T @ np.array(w_opti) - kappa * sqrt(np.array(w_opti).T @ omega @ np.array(w_opti)))
    except:
        lambd = lambd_opti
    

    w = cp.Variable(n)
    ret = np.array(mu).T @ w

    omegaEigenvalues, omegaEigenvectors = np.linalg.eig(omega)
    omegaD = np.diag(omegaEigenvalues)

    ret2 = cp.norm(np.sqrt(omegaD) @ omegaEigenvectors @ w)

    ret3 = cp.quad_form(w,sigma)


    vol_mean = cp.quad_form(w,sigma)

    better_cap = market_cap(date,n,ent)
    constraints_cap = w @ better_cap

    constraints = [cp.sum(w) == 1, w >= 0, sum(constraints_cap) == 0, vol_mean <= 0.04/365]


    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 - lambd * ret3), constraints)
    prob.solve()

    return w.value, w, prob, ret.value, ret2, ret3.value,lambd


def portfolio_contraint_lambda(date,ent,w_opti,ret_opti,ret3_opti,lambd_opti):
    sigma_mean = main.get_cov(date)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return w_opti,0,0,ret_opti,0,ret3_opti,lambd_opti
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    try:
        w_rob,w,prob,ret,ret2,ret3,lambd = weight_cp_mean_lambda(w_opti,sigma_mean,mu_mean,0.25,date,lambd_opti,ent)
    except:
        print("fail")
        return w_opti,0,0,ret_opti,0,ret3_opti,lambd_opti
    return w_rob,w,prob,ret,ret2,ret3,lambd





"""Determination des donnees sur toute la periode"""
def portfolio_construction(ent):
    w_opti,w,prob,ret,ret2,vol_mean = portfolio_contraint(str(all_dates[30]),ent)

    portfolio_over_time = []
    portfolio_return = [ret]
    portfolio_risk = [vol_mean]

    ret3 = vol_mean
    lambd = 0

    for i in range(len(all_dates[31:])):
        w_rob,w,prob,ret,ret2,ret3,lambd = portfolio_contraint_lambda(str(all_dates[31+i]),ent,w_opti,ret,ret3,lambd)
        portfolio_over_time.append(w_rob)
        portfolio_return.append(ret)
        portfolio_risk.append(ret3)
        w_opti = w_rob
    
    return portfolio_over_time,portfolio_return,portfolio_risk


#portfolio_small,portfolio_medium,portfolio_large = [portfolio_construction(75),portfolio_construction(150),portfolio_construction(300)]
portfolio_large=portfolio_construction(300)

def get_all_portfolios(portfolio):
    liste=[]
    portfolio_w,portfolio_ret,portfolio_risk = portfolio
    for i in range(len(all_dates[30:len(all_dates)-1])):
        date = all_dates[30 + i]
        print(date)
        w,ret,risk = portfolio_w[i],portfolio_ret[i],portfolio_risk[i]
        print(ret,risk)
        try:
            liste.append([str(date)[0:10],str(ret),str(risk)]+[str(x) for x in w])
        except:
            pass
    return liste
import csv
'''
import csv
with open('softConstraintsOptiSmall.csv','w',newline="") as result_file:
    wr = csv.writer(result_file,delimiter=";")
    wr.writerows(get_all_portfolios(portfolio_small))

with open('softConstraintsOptiMedium.csv','w',newline="") as result_file:
    wr = csv.writer(result_file,delimiter=";")
    wr.writerows(get_all_portfolios(portfolio_medium))
'''

with open('softConstraintsOptiLarge.csv','w',newline="") as result_file:
    wr = csv.writer(result_file,delimiter=";")
    wr.writerows(get_all_portfolios(portfolio_large))


ret_small,_,_=portfolio_small
ret_medium,_,_=portfolio_medium
ret_large,_,_=portfolio_large

#plt.plot(all_dates[30:],ret_small,all_dates[30:],ret_medium,all_dates[30:],ret_large)
#plt.show()