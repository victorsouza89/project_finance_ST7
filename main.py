import pandas as pd
import yfinance as yf
import math
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')



"""Exercise 1"""

path = "DataProjets.xlsx"

"""a)"""
df = pd.read_excel(path, sheet_name="Mapping")


def get_msft(df, sedol):
    """Prend en argument un sedol et renvoie le msft correspondant"""
    n = len(df)
    for i in range(n):
        if str(df["Sedol"][i]) == str(sedol):
            return yf.Ticker(df["Tickers"][i])



"""b)"""
df2 = pd.read_excel(path, sheet_name="MarketCaps")
df2 = df2.rename(columns={"Unnamed: 0": "date"})


"""c)"""


def nettoyage(df2):
    n = len(df2)
    columns = df2.columns
    for column in columns[1:]:
        for i in range(n - 2, -1, -1):
            if math.isnan(df2[column][i]) or df2[column][i] == 0:
                df2[column][i] = df2[column][i + 1]

    return df2

print('Nettoyage en cours')
#df2 = nettoyage(df2)

clearConsole()
print('Nettoyage terminé')
"""d)"""




def convert_date(date):
    """Prend en argument une date au format timestamp, et renvoie "year-month-day"""
    return str(date.year) + "-" + str(date.month) + "-" + str(date.day)


def get_price(msft_list, date_list):
    """Prend en argument un msft et une date au format timestamp, et renvoie un prix"""
    price_database={}
  
    for msft in msft_list:
        print(str(msft))
        todays_data = msft.history(period="20y", interval="1d")["Close"]
        pickle.dump( todays_data.index, open( "all_dates.p", "wb" )    )
        for date in todays_data.index:
            price_database[(str(msft),str(date))]= todays_data[date]

        '''
        for date in date_list:
            if list(todays_data) == []:
                price_database[(str(msft),str(date))]= 0

            keys = todays_data.index

            if date in keys:
                price_database[(str(msft),str(date))]= todays_data[date]
            else:
                n = len(keys)
                for i in range(n-1):
                    if keys[i+1] > date:
                        price_database[(str(msft),str(date))]=todays_data[keys[i]]
        '''
    return price_database

print("Creating Database...")
all_date=df2["date"]
all_msft=[get_msft(df, sedol) for sedol in df['Sedol']]
#price_database=get_price(all_msft,all_date)
#pickle.dump( price_database, open( "price.p", "wb" ) )
#price_database=pickle.load( open( "price.p", "rb" ) )
#all_dates=list(pickle.load( open( "all_dates.p", "rb" ) ))
print("Database completed")
#print(price_database)
#print(all_dates)


def get_rit(date1, date2, msft,price_database): 
    """Calcule r_i_t entre une date1 et une date 2 pour un msft"""
    P_i_t = price_database[str(msft),str(date1)]
    P_i_tbefore = price_database[str(msft),str(date2)]
    if (P_i_tbefore) == 0:
        return 0
    else:
        return P_i_t / P_i_tbefore - 1


def get_performance_indice(df, df2, price_database, t=100, N=300):
    """Calcule rt"""
    rt = 0
    all_dates=pickle.load( open( "all_dates.p", "rb" ) )
    perf=pd.read_excel("performance.csv",sep=';')
    date1, date2 = all_dates[t], all_dates[t - 1]
    columns = df2.columns[1:]
    for i in range(N):
        company = df2[columns[i]]
        w_t_i = company[t]
        msft = get_msft(df, columns[i])
        try:
            r_t_i = get_rit(date1, date2, msft,price_database)
        except:
            r_t_i=0

        rt += w_t_i * r_t_i

    return rt


date = 150
#rt = get_indice(df, df2, t=date, N=3)
#print(rt)
#rt = get_performance_indice(df, df2,price_database,t=date, N=500)
#print("indice "+str(rt))

"""Exercise 3"""

"a"


def get_average_perf(year):
    all_date = df2["date"]
    dates = []
    for i,x in enumerate(all_date):
        
        if str(x)[0:4] == year:
            dates.append(i)

    return np.mean([get_performance_indice(df, df2, price_database,t=x, N=500) for x in dates])


#print("moyenne " +str(get_average_perf("2021")))

def get_deviation_perf(year):
    all_date = df2["date"]
    dates = []
    for i,x in enumerate(all_date):
        
        if str(x)[0:4] == year:
            dates.append(i)

    return np.std([get_performance_indice(df, df2,price_database, t=x, N=500) for x in dates])

#print("deviation "+str(get_deviation_perf("2021")))

def get_indice(perf_list):
    p_0=1
    indice=[p_0]
    for x in perf_list:
        indice.append(indice[-1]*x+1)
    return indice

#all_rt=[get_performance_indice(df, df2,price_database,t, N=100) for t in range(1,100,5)]
#plt.plot(all_rt)
#plt.show()

def get_all_rti(N,price_database,df2):
    all_dates=pickle.load( open( "all_dates.p", "rb" ) )
    d={"Dates" : all_dates[1:]}
    columns = df2.columns[1:]
    for i in range(N):
        liste=[]
        msft,sedol=get_msft(df, columns[i]),columns[i]
        for t in range(1,len(all_dates)):
            date1, date2 = all_dates[t], all_dates[t - 1]
            
            try:

                r_t_i = get_rit(date1, date2, msft,price_database)
            except:
                r_t_i=0
            liste.append(r_t_i)
        d[sedol]=liste
    return d

def get_cov(date):
    all_dates=pickle.load( open( "all_dates.p", "rb" ) )
    t=0
    for i in range(len(all_dates)):
        if all_dates[i]==date:
            t=i
    if t<=600:
        return [[0]]
    perf=pd.read_csv('performance.csv',sep=';')
    columns = df2.columns[1:]
    data=[]
    for i in range(len(columns)):
            sedol=columns[i]
            liste=perf[str(sedol)][t-600:t]
            data.append(liste)
    return(np.cov(data))
            





def get_all_indicators(df2):
    dates=df2["date"]
    all_dates=pickle.load( open( "all_dates.p", "rb" ) )
    perf=pd.read_csv('performance.csv',sep=';')
    r={}
    for j,date in enumerate(dates):
        t=0
        for i in range(len(all_dates)):
            if all_dates[i]==date:
                t=i
        columns = df2.columns[1:]
        rt=0
        for i in range(len(columns)):
            company = df2[columns[i]]
            w_t_i = company[j]
            sedol=columns[i]
            r_t_i = perf[str(sedol)][t]
            rt+=w_t_i*r_t_i
        r[date]=[rt]

    return r

def get_all_indicators2(df2):
    dates=df2["date"]
    all_dates=pickle.load( open( "all_dates.p", "rb" ) )
    perf=pd.read_csv('performance.csv',sep=';')
    risque={}
    for j,date in enumerate(dates):
        print(date)
        t=0
        for i in range(len(all_dates)):
            if all_dates[i]==date:
                t=i
        columns = df2.columns[1:]
        risquet=0
        covariance=get_cov(date)
        if covariance[0][0]==0:
            risquet=0
        else:
            for a in range(len(columns)):
                for b in range(len(columns)):
                    risquet+=covariance[a,b]*df2[columns[a]][j]*df2[columns[b]][j]
        risque[date]=[risquet]
     

    return risque

df_rit = pd.DataFrame(data=get_all_indicators2(df2))
df_rit.to_csv("indicators2.csv",sep=';')
           
        
        


'''
df_rit = pd.DataFrame(data=get_all_rti(len(df2.columns[1:]),price_database,df2))
print(df_rit)
df_rit.to_csv("performance.csv",sep=';',index=False)
'''