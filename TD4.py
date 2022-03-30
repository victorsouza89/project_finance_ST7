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
all_dates=pickle.load( open( "all_dates.p", "rb" ) )
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
    res = np.zeros((1,n))
    for i in range(n):
        daily_data_cap[i] = data_cap[i]
    sorted_data = np.sort(daily_data_cap)
    boundary = sorted_data[-ent]

    for i in range(n):
        if daily_data_cap[i] < boundary:
            res[0,i] = 1
    return res


def weight_cp_mean(sigma, mu, kappa, date, ent):

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

    #better_cap = market_cap(date,n,ent)
    #cap_constraint = np.diag(w @ better_cap)

    constraints = [cp.sum(w) == 1, w >= 0, vol_mean <= 0.04]

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 ), constraints)
    prob.solve()

    return w.value, w, prob, ret, ret2



def portfolio_contraint(date,ent):
    sigma_mean = main.get_cov(date)
    print(sigma_mean)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return 0,0,0,0
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    w_rob,w,prob,ret,ret2 = weight_cp_mean(sigma_mean,mu_mean,0.25,date,ent)
    return w_rob,w,prob,ret,ret2


"""Partie 2 : lambda equivalent"""

def weight_cp_mean_lambda(sigma, mu, kappa, date,ent=150):
    date=date[0:10]
    n = len(sigma[0])
    m = len(sigma)
    

    omega = np.zeros((m, m))
    for i in range(m):
        omega[i][i] = sigma[i][i]


    one = np.ones((n,1))
    lambd = - 1 / (one.T @ inv((sigma+0.000001*np.identity(n))) @ one)
    lambd = lambd[0][0]
    print(lambd)

    w = cp.Variable(n)
    ret = np.array(mu).T @ w
    print(ret.shape)

    omegaEigenvalues, omegaEigenvectors = np.linalg.eig(omega)
    omegaD = np.diag(omegaEigenvalues)

    ret2 = cp.norm(np.sqrt(omegaD) @ omegaEigenvectors @ w)
    print(ret2.shape)


    #better_cap = market_cap(date,n,ent)
    
    #constraints_cap = np.array([w[i] for i in range(len(better_cap[0])) if better_cap[0,i]==1])

    #constraints = [cp.sum(w) == 1, w >= 0, constraints_cap == np.array([0]*len(constraints_cap))]
    

    print(w.shape)
    #nv_liste = np.array(w).reshape(-1,1)
    #print(nv_liste.shape)
    #constraints_cap = np.diag(nv_liste @ better_cap)


    #constraints = [cp.sum(w) == 1, w >= 0, sum(constraints_cap @ w) == 0]
    
    constraints = [cp.sum(w) == 1, w >= 0]

    ret3 = cp.quad_form(w,sigma)
    print(ret3.shape)
    print(lambd.shape)

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2), constraints)
    prob.solve()

    return w.value, w, prob, ret, ret2


def portfolio_contraint_lambda(date,ent):
    
    sigma_mean = main.get_cov(date)
    print(sigma_mean)
    if sigma_mean[0][0] == 0:
        print(sigma_mean)
        return 0,0,0,0
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    w_rob,w,prob,ret,ret2 = weight_cp_mean_lambda(sigma_mean,mu_mean,0.25,date,ent)
    return w_rob,w,prob,ret,ret2



"""Partie 3 : optimisation pour les 75 et les 300 plus grades capitalisations"""


print(portfolio_contraint_lambda("2019-05-31 00:00:00",300))