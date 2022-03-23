import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cvxpy as cp
import pandas as pd
from copy import deepcopy
import main
from math import sqrt

volatility = np.array([[14.3, 17.4, 21.2, 4.3, 4, 8.4, 0.5]])
mu = np.array([6, 7, 9.5, 1.5, 1.3, 3.2, 0])
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
    df = pd.read_csv(path)

    data_cap = df.loc[:,[date]]

    daily_data_cap = np.zeros(n)
    res = np.zeros(1,n)
    for i in range(n):
        daily_data_cap[i] = data_cap[i]
    sorted_data = np.sort(daily_data_cap)
    boundary = sorted_data[-ent]

    for i in range(n):
        if daily_data_cap[i] < boundary:
            res[i] = 1
    
    return res


def weight_cp_mean(sigma, mu, kappa, date):

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

    vol_mean = w.T @ sigma @ w

    better_cap = market_cap(date,n,ent)
    cap_constraint = np.diag(w @ better_cap)

    constraints = [cp.sum(w) == 1, w >= 0, vol_mean <= 0.04, cap_constraint == np.zeros(n,n)]

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 ), constraints)
    prob.solve()

    return w.value, w, prob, ret, ret2



def portfolio_contraint(date):
    date_cov = date + " 00:00:00"
    sigma_mean = main.get_cov(date_cov)
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    w_rob,w,prob,ret,ret2 = weight_cp_mean(sigma_mean,mu_mean,0.25,date)
    return w_rob,w,prob,ret,ret2


"""Partie 2 : lambda equivalent"""

def weight_cp_mean_lambda(sigma, mu, kappa, date,ent=150):
    
    n = len(sigma[0])
    m = len(sigma)

    omega = np.zeros((m, m))
    for i in range(m):
        omega[i][i] = sigma[i][i]


    one = np.ones((n,1))
    lambd = - 1 / (one.T @ sigma.inv() @ one)

    w = cp.Variable(n)
    ret = np.array(mu).T @ w

    omegaEigenvalues, omegaEigenvectors = np.linalg.eig(omega)
    omegaD = np.diag(omegaEigenvalues)

    ret2 = cp.norm(np.sqrt(omegaD) @ omegaEigenvectors @ w)

    better_cap = market_cap(date,n,ent)
    cap_constraint = np.diag(w @ better_cap)

    constraints = [cp.sum(w) == 1, w >= 0, cap_constraint == np.zeros(n)]

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2  - lambd * w.T @ sigma @ w), constraints)
    prob.solve()

    return w.value, w, prob, ret, ret2


def portfolio_contraint_lambda(date,ent):
    date_cov = date + " 00:00:00"
    sigma_mean = main.get_cov(date_cov)
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

print(portfolio_contraint_lambda("2017-05-31",75))
print(portfolio_contraint_lambda("2017-05-31",300))


w_prime = np.array([[x] for x in w])