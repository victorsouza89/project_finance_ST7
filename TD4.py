import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cvxpy as cp
import pandas as pd
from copy import deepcopy
import main

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

def weight_cp_mean(sigma, mu, kappa, vol):

    n = len(sigma[0])
    m = len(sigma)
    omega = deepcopy(sigma)

    omega = np.zeros((m, m))
    for i in range(m):
        omega[i][i] = sigma[i][i]


    w = cp.Variable(n)
    ret = np.array(mu).T @ w

    omegaEigenvalues, omegaEigenvectors = np.linalg.eig(omega)
    omegaD = np.diag(omegaEigenvalues)

    ret2 = cp.norm(np.sqrt(omegaD) @ omegaEigenvectors @ w)

    risk = cp.quad_form(w, sigma)

    vol_mean = w.T @ sigma @ w

    constraints = [cp.sum(w) == 1, w >= 0,risk<=vol, vol_mean <= 0.04]

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 ), constraints)
    prob.solve()
    return w.value, w, prob, risk, ret, ret2



def portfolio_contraint():
    sigma_mean = main.get_cov(date,all_dates=all_dates,perf=perf)
    mu_mean = np.zeros(len(sigma_mean))
    for i in range(len(sigma_mean)):
        mu_mean[i] = 0.5 * sqrt(sigma_mean[i][i])
    w_rob,w,prob,risk,ret,ret2 = weight_cp_mean(sigma_mean,mu_mean,0.25,vol)
