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


"Exercice 2"

def get_mu(sigma):
    mu = np.array(len(sigma[0]))
    for j in range(len(sigma[0])):
        for i in range(len(sigma)):
            mu[j] += 1/(len(sigma)) * sigma[i][j]
    return mu


def market_cap(date,n,ent=10):
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


def weight_cp_mean_tc(sigma, mu, kappa, date, ent):
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



def portfolio_contraint_tc(date,ent):
    sigma_mean = main.get_cov(date)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return 0,0,0,0,0
    mu_mean = get_mu(sigma_mean)
    w_rob,w,prob,ret,ret2,vol_mean = weight_cp_mean_tc(sigma_mean,mu_mean,0.5,date,ent)
    return w_rob,w,prob,ret,ret2, vol_mean



"""Partie 2 : lambda equivalent a detendre"""

def weight_cp_mean_tc(w_opti, sigma, mu, kappa, date, psi=0.02, ent=10):
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

    ret3 = cp.norm(w - w_opti)


    vol_mean = cp.quad_form(w,sigma)

    better_cap = market_cap(date,n,ent)
    constraints_cap = w @ better_cap

    constraints = [cp.sum(w) == 1, w >= 0, sum(constraints_cap) == 0, vol_mean <= 0.04/365]


    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 - psi * ret3), constraints)
    prob.solve()

    return w.value, w, prob, ret.value, ret2, ret3.value


def portfolio_contraint_tc_psi(date,ent,w_opti,ret_opti,ret3_opti,psi):
    sigma_mean = main.get_cov(date)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return w_opti,0,0,ret_opti,0,ret3_opti
    mu_mean = get_mu(sigma_mean)
    try:
        w_rob,w,prob,ret,ret2,ret3 = weight_cp_mean_tc(w_opti,sigma_mean,mu_mean,0.5,date,psi,ent)
    except:
        print("fail")
        return w_opti,0,0,ret_opti,0,ret3_opti
    return w_rob,w,prob,ret,ret2,ret3





"""Determination des donnees sur toute la periode"""
def portfolio_construction_tc(ent=10,psi=0.02):
    w_opti,w,prob,ret,ret2,vol_mean = portfolio_contraint_tc(str(all_dates[30]),ent)

    portfolio_over_time = []
    portfolio_return = [ret]
    portfolio_risk = [vol_mean]

    ret3 = vol_mean

    for i in range(len(all_dates[31:])):
        w_rob,w,prob,ret,ret2,ret3 = portfolio_contraint_tc_psi(str(all_dates[31+i]),ent,w_opti,ret,ret3,psi)
        portfolio_over_time.append(w_rob)
        portfolio_return.append(ret)
        portfolio_risk.append(ret3)
        w_opti = w_rob
    
    return portfolio_over_time,portfolio_return,portfolio_risk


portfolio_value = portfolio_construction_tc(10,0.02)


def get_portfolios(portfolio):
    liste=[]
    portfolio_w,portfolio_ret,portfolio_risk = portfolio
    for i in range(len(all_dates[30:len(all_dates)-1])):
        date = all_dates[30 + i]
        print(date)
        w,ret,risk = portfolio_w[i],portfolio_ret[i],portfolio_risk[i]
        liste.append([str(date)[0:10],str(ret),str(risk)]+[str(x) for x in w])
    return liste

import csv
with open('transactionCost.csv','w',newline="") as result_file:
    wr = csv.writer(result_file,delimiter=";")
    wr.writerows(get_portfolios(portfolio_value))

