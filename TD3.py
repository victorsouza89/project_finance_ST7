import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cvxpy as cp
import pandas as pd
from copy import deepcopy


"""ex1"""
"""a"""

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
lambd = 4
kappa = 0.2


def get_sigma(rho, volatility):
    n = len(volatility[0])
    sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sigma[i, j] = rho[i, j] * volatility[0, i] * volatility[0, j]

    for i in range(n):
        for j in range(i):
            sigma[i, j] = sigma[j, i]
    return sigma


def weight_MVO(volatility=volatility, sigma = get_sigma(rho, volatility)):    
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return inv_sigma.dot(mu) / ones.T.dot(inv_sigma).dot(mu)


print("weight_MVO")
print(weight_MVO())


def weight_EW(volatility=volatility, sigma = get_sigma(rho, volatility)):    
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (I.dot(ones)).T / (ones.T.dot(I).dot(ones))


print("-----")
print("weight_EW")
print(weight_EW())


def weight_IV(volatility=volatility, sigma = get_sigma(rho, volatility)):
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (Lambda ** 2) / ones.T.dot((Lambda ** -2))


print("-----")
print("weight_IV")
print(weight_IV())


def weight_ERB(volatility=volatility, sigma = get_sigma(rho, volatility)):    
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (Lambda) / ones.T.dot((Lambda ** -1))


print("-----")
print("weight_ERB")
print(weight_ERB())


def weight_MV(volatility=volatility, sigma = get_sigma(rho, volatility)):    
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (inv_sigma.dot(ones)).T / ones.T.dot(inv_sigma).dot(ones)


print("-----")
print("weight_MV")
print(weight_MV())


"""Exo 2"""
"""a"""


def weight_cp(sigma, mu, lambd, kappa, mode="sigma"):
    """La fonction ne marche pas car j'ai une erreur : "cvxpy.error.DCPError: Problem does not follow DCP rules. Specifically:
    The objective is not DCP. Its following subexpressions are not:"

    La variable mode peut prendre les valeurs sigma, diag, ou identite, en fonction de la valeur qu'on veut pour la matrice omega
    """

    n = len(sigma[0])
    m = len(sigma)
    omega = deepcopy(sigma)

    if mode == "diag":
        omega = np.zeros((m, m))
        for i in range(m):
            omega[i][i] = sigma[i][i]

    elif mode == "identite":
        omega = np.zeros((m, m,))
        for i in range(m):
            omega[i][i] = 1

    w = cp.Variable(n)
    ret = np.array(mu).T @ w

    omegaEigenvalues, omegaEigenvectors = np.linalg.eig(omega)
    omegaD = np.diag(omegaEigenvalues)

    ret2 = cp.norm(np.sqrt(omegaD) @ omegaEigenvectors @ w)

    risk = cp.quad_form(w, sigma)

    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 - lambd / 2 * risk), constraints)
    prob.solve()
    return w.value, w, prob, risk, ret, ret2


print("-----")
print("weight_CP")
print(
    weight_cp(
        sigma=get_sigma(rho=rho, volatility=volatility), mu=mu, lambd=lambd, kappa=kappa, mode="sigma"
    )[0]
)

"""b"""
print("-----")
print("kappa = 10000")
print(weight_cp(sigma=get_sigma(rho=rho, volatility=volatility), mu=mu, lambd=lambd, kappa=10000, mode='sigma')[0])


"""Exercice 3"""
"""a"""
print("-----")
print("omega diag")
print(weight_cp(sigma=get_sigma(rho=rho, volatility=volatility), mu=mu, lambd=lambd, kappa=kappa, mode='diag')[0])
print("-----")
print("omega identity")
print(weight_cp(sigma=get_sigma(rho=rho, volatility=volatility), mu=mu, lambd=lambd, kappa=kappa, mode='identite')[0])


"""b"""
print("-----")
print("omega diag, kappa = 1000")
print(weight_cp(sigma=get_sigma(rho=rho, volatility=volatility), mu=mu, lambd=lambd, kappa=10000, mode='diag')[0])
print("-----")
print("omega identity, kappa = 1000")
print(weight_cp(sigma=get_sigma(rho=rho, volatility=volatility), mu=mu, lambd=lambd, kappa=10000, mode='identite')[0])

"""Exercice 4"""

perf_list=pd.read_csv('performance.csv',sep=';')
def mu_estimate(date,perf_list=perf_list):
    sedol_list=perf_list.keys()[1:]
    dates=perf_list['Dates']
    j=0
    for i in range(len(dates)):
        if str(dates[i]+' 00:00:00')==str(date):
            j=i
    if j>=600:
        return [np.mean([perf_list[sedol][i] for i in range(j-600,j) ]) for sedol in   sedol_list  ]
    else:
        return [0 for _ in range(len(sedol_list))]

print("-----")
import main
def sigma_estimate(date):
    return main.get_cov(date)

def get_weight(date,volatility, weight):
    mu=mu_estimate(date)
    n=len(mu)
    sigma=sigma_estimate(date)
    if sigma[0][0]==0:
        return [0 for _ in range(n)],0,0
            
    _, w, prob, risk, ret, ret2 = weight_cp(sigma=sigma, mu=mu, lambd=lambd, kappa=kappa, mode='diag')
    
    return w.value, prob.value, risk.value, ret.value, ret2.value 
print("-----")
path = "DataProjets.xlsx"
df2 = pd.read_excel(path, sheet_name="MarketCaps")
df2 = df2.rename(columns={"Unnamed: 0": "date"})
base=pd.read_csv('indicators.csv',sep=';').T
base.columns = base.iloc[0]
base = base.drop(['date'])
base.index = pd.to_datetime(base.index)

def get_all_weights(df2, weight, base=base):
    all_dates=df2['date']
    liste=[]
    for date in pd.to_datetime(all_dates):
        print(date)
        if True:
            volatility=base.loc[date,'volatility']
            all_returned = get_weight(date,volatility, weight)
            print("a")
            #liste.append([str(date)[0:10],str(perf),str(risk)]+[str(x) for x in w])
            print(all_returned)
        else:
            print('fail')
    return liste

liste=get_all_weights(df2, lambda sigma, mu : weight_cp(sigma=sigma, mu=mu, lambd=lambd, kappa=kappa, mode='diag'))
import csv
with open('outputCP.csv','w',newline="") as result_file:
    wr = csv.writer(result_file,delimiter=";")
    wr.writerows(liste)
