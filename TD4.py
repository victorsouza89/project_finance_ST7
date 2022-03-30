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
    df = pd.read_csv(path,index_col=0)

    data_cap = df.loc[date]

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

    return w.value, w, prob, ret, ret2



def portfolio_contraint(date,ent):
    sigma_mean = main.get_cov(date)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return 0,0,0,0,0
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    w_rob,w,prob,ret,ret2 = weight_cp_mean(sigma_mean,mu_mean,0.25,date,ent)
    return w_rob,w,prob,ret,ret2



"""Partie 2 : lambda equivalent a detendre"""

def weight_cp_mean_lambda(w_opti, sigma, mu, kappa, date,ent=150):
    date=date[0:10]

    n = len(sigma[0])
    m = len(sigma)
    

    omega = np.zeros((m, m))
    for i in range(m):
        omega[i][i] = sigma[i][i]

    lambd = (np.array(w_opti).T @ sigma @ np.array(w_opti))**(-1) * (np.array(mu).T @ np.array(w_opti) - kappa * sqrt(np.array(w_opti).T @ omega @ np.array(w_opti)))

    w = cp.Variable(n)
    ret = np.array(mu).T @ w

    omegaEigenvalues, omegaEigenvectors = np.linalg.eig(omega)
    omegaD = np.diag(omegaEigenvalues)

    ret2 = cp.norm(np.sqrt(omegaD) @ omegaEigenvectors @ w)

    ret3 = cp.quad_form(w,sigma)


    better_cap = market_cap(date,n,ent)
    constraints_cap = w @ better_cap

    constraints = [cp.sum(w) == 1, w >= 0, sum(constraints_cap) == 0]


    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 - lambd * ret3), constraints)
    prob.solve()

    return w.value, w, prob, ret, ret2, ret3


def portfolio_contraint_lambda(date,ent,w_opti):
    
    sigma_mean = main.get_cov(date)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return 0,0,0,0,0,0
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    w_rob,w,prob,ret,ret2,ret3 = weight_cp_mean_lambda(w_opti,sigma_mean,mu_mean,0.25,date,ent)
    return w_rob,w,prob,ret,ret2,ret3




"""Partie 3 : optimisation pour les 75 et les 300 plus grades capitalisations"""



"""Determination des donnees sur toute la periode"""
w_opti,w,prob,ret,ret2 = portfolio_contraint(str(all_dates[30]),150)
portfolio_over_time = []
portfolio_return = [ret]

for x in all_dates[31:]:
    try:
        a,b,c,d,e,f = portfolio_contraint_lambda(str(x),150,w_opti)
        w_opti,ret = a,d
    except:
        pass
    portfolio_over_time.append(w_opti)
    portfolio_return.append(ret)


plt.plot(all_dates[30:],portfolio_return)
plt.show()