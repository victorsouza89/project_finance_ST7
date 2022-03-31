import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd

path = "DataProjets.xlsx"
perf=pd.read_csv('performance.csv',sep=';')
df2 = pd.read_excel(path, sheet_name="MarketCaps")
df2 = df2.rename(columns={"Unnamed: 0": "date"})
all_dates = pd.to_datetime(df2['date'])

base=pd.read_csv('indicators.csv',sep=';').T
base.columns = base.iloc[0]
base = base.drop(['date'])
base.index = pd.to_datetime(base.index, format='%d/%m/%Y', errors='coerce')


def sigma_estimate(date):
    return get_cov(date)

def get_cov(date,all_dates=all_dates,perf=perf,lg=12):

    t=0
    for i in range(len(all_dates)):
        if all_dates[i]==date:
            t=i
    if t<=lg:
        return [[0]]

    columns = df2.columns[1:]
    data=[]
    for i in range(len(columns)):
            sedol=columns[i]
            liste=perf[str(sedol)][t-lg:t]
            data.append(liste)
    return(np.cov(data))

def get_all_weights(df2, weight, base=base):
    all_dates=df2['date']
    liste=pd.Series()
    for date in pd.to_datetime(all_dates):
        print(date)
        try:
            volatility=base.loc[date,'daily volatility (variance)']
            w = get_weight(date,volatility, weight)
            liste[date] = w
            #print(w)
        except:
            print('fail')
    return liste

def get_weight(date,volatility, weight):
    sigma=sigma_estimate(date)
    n = len(sigma[0])
    sigma=sigma+0.000001*np.identity(n)
    
    return weight(volatility, sigma) 


def weight_EW(volatility, sigma):    
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (I.dot(ones)).T / (ones.T.dot(I).dot(ones))


print("-----")
print("weight_EW")
liste = get_all_weights(df2, weight_EW)
print(liste)
liste.to_csv('outputEW.csv')


def weight_IV(volatility, sigma):
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = np.diag(sigma.diagonal())
    inv_lambda2 = inv(Lambda**2)
    return (inv_lambda2.dot(ones)/ones.T.dot(inv_lambda2.dot(ones))).T


print("-----")
print("weight_IV")
liste = get_all_weights(df2, weight_IV)
print(liste)
liste.to_csv('outputIV.csv')



def weight_ERB(volatility, sigma):    
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = np.diag(sigma.diagonal())
    inv_lambda = inv(Lambda)
    return (inv_lambda.dot(ones)/ones.T.dot(inv_lambda.dot(ones))).T


print("-----")
print("weight_ERB")
liste = get_all_weights(df2, weight_ERB)
print(liste)
liste.to_csv('outputERB.csv')


def weight_MV(volatility, sigma):    
    N = np.shape(sigma)[0]
    ones = np.ones((N, 1))
    I = np.eye(N)
    inv_sigma = inv(sigma)
    Lambda = sigma.diagonal()
    return (inv_sigma.dot(ones)).T / ones.T.dot(inv_sigma).dot(ones)


print("-----")
print("weight_MV")
liste = get_all_weights(df2, weight_MV)
print(liste)
liste.to_csv('outputMV.csv')