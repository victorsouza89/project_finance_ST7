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
all_dates=pickle.load( open( "all_dates.p", "rb" ) )
perf=pd.read_csv('performance.csv',sep=';')


def get_cov(date,all_dates=all_dates,perf=perf,lg=600):

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



perf_list=pd.read_csv('performance.csv',sep=';')
def mu_estimate(date,perf_list=perf_list,lg=600):

    sedol_list=perf_list.keys()[1:]
    dates=perf_list['Dates']
    j=0
    for i in range(len(dates)):
        if str(dates[i]+' 00:00:00')==str(date):
            j=i
    if j>=lg:
        return [np.mean([perf_list[sedol][i] for i in range(j-lg,j) ]) for sedol in   sedol_list  ]
    else:
        return [0 for _ in range(len(sedol_list))]

def get_all_cov():
    all_dates=df2['date']
    dic={}
    for date in all_dates:
        print(date)
        dic[str(date)]=[get_cov(date)]
    return dic


#pickle.dump( get_all_cov(), open( "all_cov.p", "wb" ) )



def get_all_indicators(df2):
    dates=df2["date"]
    #all_dates=pickle.load( open( "all_dates.p", "rb" ) )
    #perf=pd.read_csv('performance.csv',sep=';')
    r={}
    for j,date in enumerate(dates):
        #print(date)
        mu=mu_estimate(date,lg=30)
        columns = df2.columns[1:]

        #for i in range(len(all_dates)):
        #    if all_dates[i]==date:
        #        t=i
        rt=0
        for i in range(len(columns)):
            company = df2[columns[i]]
            w_t_i = company[j]
            r_t_i = mu[i]
            rt+=w_t_i*r_t_i
        print(rt)
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
        covariance=get_cov(date,lg=30)
        if covariance[0][0]==0:
            risquet=0
        else:
            for a in range(len(columns)):
                for b in range(len(columns)):
                    risquet+=covariance[a,b]*df2[columns[a]][j]*df2[columns[b]][j]
        risque[date]=[risquet]
     

    return risque

#df_rit = pd.DataFrame(data=get_all_indicators2(df2))
#df_rit.to_csv("indicators3.csv",sep=';')
#weights=pd.read_csv('indicators.csv',sep=';')
#all_dates=df2['date']
#print(weights)
def get_maximum_drawdown(date,w):
    t=0
    for (i,d) in enumerate(all_dates):
        if str(d)[0:10]==date:
            t=i
    if t<24:
        return 0
    
    min=1
    max=1
    current=1
    for i in range(0,t):
        perf=weights[str(all_dates[i])[0:10]][w]

        if i<t-24:
            current=current*(1+perf)
            min=current
            max=current
        else:
            
            current=current*(1+perf)
            #print(current)
            min=np.min([min,current])
            max=np.max([max,current])
    return (max-min)/max

def get_all_maximum_drawdown():
    l1=[]
    l2=[]
    for date in all_dates:
        l1.append(get_maximum_drawdown(str(date)[0:10],0 ))
        l2.append(get_maximum_drawdown(str(date)[0:10],4 ))
        
        
    return l1,l2
'''
l1,l2=get_all_maximum_drawdown()
plt.plot(l1)
plt.plot(l2)
plt.show()
'''



           
df3 = pd.read_excel(path, sheet_name="Sector", index_col=0)
def sector_sorting(date):
    sector_sorted=[[] for _ in range(16)]
    liste=df3.loc[date]
    for c in liste.index:
        sector_sorted[int(liste[c])].append(c)
    return sector_sorted

def mu_estimate_sector(date,companies,perf_list=perf_list,lg=600):
    dates=perf_list['Dates']
    j=0
    for i in range(len(dates)):
        if str(dates[i]+' 00:00:00')==str(date):
            j=i
    if j>=lg:
        return [np.mean([perf_list[str(sedol)][i] for i in range(j-lg,j) ]) for sedol in   companies  ]
    else:
        return [0 for _ in range(len(companies))]

def NormalizeData(data):
    tot=np.sum(data)
    return 1/tot*data

def performance_sector(sector):
    dates=df2["date"]
    columns = df2.columns[1:]
    r={}
    for j,date in enumerate(dates):
        try:
            companies_sector=sector_sorting(date)[sector]
            mu=mu_estimate_sector(date,companies_sector,lg=30)
            rt=0
        
            for (i,c) in enumerate(companies_sector):
                company_weight = NormalizeData(df2[c])
               
                w_t_i = company_weight[j]
                r_t_i = mu[i]
                rt+=w_t_i*r_t_i
            print(rt)
            r[date]=[rt]

        except:
            print('fail')
            r[date]=[0]
        
    return r
performance_sector(14)







'''
df_rit = pd.DataFrame(data=get_all_rti(len(df2.columns[1:]),price_database,df2))
print(df_rit)
df_rit.to_csv("performance.csv",sep=';',index=False)
'''