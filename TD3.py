import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from numpy.linalg import inv
#import cvxpy as cp

"""ex1"""
'''a'''

volatility = np.array([[14.3,17.4,21.2,4.3,4,8.4,0.5]])
mu = np.array([6,7,9.5,1.5,1.3,3.2,0])
rho =  np.array([[1,0.82,0.78,0.1,0,0.5,0],[0.82,1,0.85,0.12,0.08,0.63,0],[0.78,0.85,1,0.05,0.03,0.71,0],[0.1,0.12,0.05,1,0.65,0.2,0],[0,0.08,0.03,0.65,1,0.23,0],[0.5,0.63,0.71,0.2,0.23,1,0],[0,0,0,0,0,0,1]])

def get_sigma(rho,volatility):
    n=len(volatility[0])
    sigma=np.zeros((n,n))
    for i in range(n):
        for j in range(n):

            sigma[i,j]=rho[i,j]*volatility[0,i]*volatility[0,j]
    return sigma


def weight_MVO(volatility=volatility,mu=mu,rho=rho):
    sigma=get_sigma(rho,volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return inv_sigma.dot(mu) / ones.T.dot(inv_sigma).dot(mu)

print("weight_MVO")
print(weight_MVO())

def weight_EW(volatility=volatility,mu=mu,rho=rho):
    sigma=get_sigma(rho,volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (I.dot(ones)).T / (ones.T.dot(I).dot(ones))

print("-----")
print("weight_EW")
print(weight_EW())

def weight_IV(volatility=volatility,mu=mu,rho=rho):
    sigma=get_sigma(rho,volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (Lambda**2) / ones.T.dot((Lambda**-2))

print("-----")
print("weight_IV")
print(weight_IV())


def weight_ERB(volatility=volatility,mu=mu,rho=rho):
    sigma=get_sigma(rho,volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (Lambda) / ones.T.dot((Lambda**-1))

print("-----")
print("weight_ERB")
print(weight_ERB())

def weight_MV(volatility=volatility,mu=mu,rho=rho):
    sigma=get_sigma(rho,volatility)
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (inv_sigma.dot(ones)).T / ones.T.dot(inv_sigma).dot(ones)

print("-----")
print("weight_MV")
print(weight_MV())