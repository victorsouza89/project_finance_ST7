import numpy as np

np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cvxpy as cp
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


def weight_MVO(volatility=volatility, mu=mu, rho=rho):
    sigma = get_sigma(rho, volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return inv_sigma.dot(mu) / ones.T.dot(inv_sigma).dot(mu)


print("weight_MVO")
print(weight_MVO())


def weight_EW(volatility=volatility, mu=mu, rho=rho):
    sigma = get_sigma(rho, volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (I.dot(ones)).T / (ones.T.dot(I).dot(ones))


print("-----")
print("weight_EW")
print(weight_EW())


def weight_IV(volatility=volatility, mu=mu, rho=rho):
    sigma = get_sigma(rho, volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (Lambda ** 2) / ones.T.dot((Lambda ** -2))


print("-----")
print("weight_IV")
print(weight_IV())


def weight_ERB(volatility=volatility, mu=mu, rho=rho):
    sigma = get_sigma(rho, volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (Lambda) / ones.T.dot((Lambda ** -1))


print("-----")
print("weight_ERB")
print(weight_ERB())


def weight_MV(volatility=volatility, mu=mu, rho=rho):
    sigma = get_sigma(rho, volatility)
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


def weight_cp(rho, volatility, mu, lambd, kappa, mode="sigma"):
    """La fonction ne marche pas car j'ai une erreur : "cvxpy.error.DCPError: Problem does not follow DCP rules. Specifically:
    The objective is not DCP. Its following subexpressions are not:"

    La variable mode peut prendre les valeurs sigma, diag, ou identite, en fonction de la valeur qu'on veut pour la matrice omega
    """

    n = len(volatility[0])
    sigma = get_sigma(rho=rho, volatility=volatility)
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
    ret = mu @ w

    ret2 = cp.sqrt(cp.quad_form(w, omega))

    risk = cp.quad_form(w, sigma)

    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(cp.Maximize(ret - kappa * ret2 - lambd / 2 * risk), constraints)
    prob.solve()
    return w.value


print(
    weight_cp(
        rho=rho, volatility=volatility, mu=mu, lambd=lambd, kappa=kappa, mode="sigma"
    )
)

"""b"""
# print(weight_cp(rho=rho, volatility=volatility, mu=mu, lambd=lambd, kappa=10000, mode='sigma'))


"""Exercice 3"""
"""a"""
# print(weight_cp(rho=rho, volatility=volatility, mu=mu, lambd=lambd, kappa=kappa, mode='diag'))
# print(weight_cp(rho=rho, volatility=volatility, mu=mu, lambd=lambd, kappa=kappa, mode='identite'))


"""b"""
# print(weight_cp(rho=rho, volatility=volatility, mu=mu, lambd=lambd, kappa=10000, mode='diag'))
# print(weight_cp(rho=rho, volatility=volatility, mu=mu, lambd=lambd, kappa=10000, mode='identite'))
