import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cvxpy as cp

"""ex1"""
'a'
volatility = np.array([[14.3,17.4,21.2,4.3,4,8.4,0.5]])
mu = np.array([[6,7,9.5,1.5,1.3,3.2,0]])
rho =  np.array([[1,0.82,0.78,0.1,0,0.5,0],[0.82,1,0.85,0.12,0.08,0.63,0],[0.78,0.85,1,0.05,0.03,0.71,0],[0.1,0.12,0.05,1,0.65,0.2,0],[0,0.08,0.03,0.65,1,0.23,0],[0.5,0.63,0.71,0.2,0.23,1,0],[0,0,0,0,0,0,1]])


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
            v+=weight[i]*weight[j]*rho[i,j]
    return v

w=weight()
print("weight")
print(w)  
print("volatility")
print(get_volatility(w,rho))

'c'
def weight_cp():
    n=len(volatility[0])
    delta=get_delta(rho,volatility)
    w = cp.Variable(n)
    gamma = 0.5
    ret = mu@w 
    risk = cp.quad_form(w, delta)
    prob = cp.Problem(cp.Maximize(ret - gamma*risk))
    prob.solve()
    return w.value

print(weight_cp())


"""ex2"""


