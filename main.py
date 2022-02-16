import pandas as pd
import yfinance as yf
import math

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


df2 = nettoyage(df2)


"""d)"""

def get_price(date,share):
    msft = yf.Ticker(share)  
    todays_data = msft.history(period='20y')
    return todays_data["Close"][date]

def convert_date(date):
    """Prend en argument une date au format timestamp, et renvoie "year-month-day""""
    return str(date.year) + "-" + str(date.month) + "-" + str(date.day)


def get_price(msft, date):
    """Prend en argument un msft et une date au format timestamp, et renvoie un prix"""
    todays_data = msft.history(period="20y")["Close"]
    todays_data = dict(todays_data)
    keys = list(todays_data.keys())
    if date in keys:
        return todays_data[date]
    else:
        n = len(keys)
        for i in range(n):
            if keys[i] < date and keys[i + 1] > date:
                return todays_data[keys[i]]


def get_rit(date1, date2, msft): 
    """Calcule r_i_t entre une date1 et une date 2 pour un msft"""
    P_i_t = get_price(msft, date1)
    P_i_tbefore = get_price(msft, date2)
    if (P_i_tbefore) == 0:
        return -1
    else:
        return P_i_t / P_i_tbefore - 1


def get_indice(df, df2, t=100, N=300):
    """Calcule rt"""
    rt = 0
    date1, date2 = df2["date"][t], df2["date"][t - 1]
    columns = df2.columns[1:]
    for i in range(N):
        company = df2[columns[i]]

        print("test2")
        w_t_i = company[t]

        print("test3")

        msft = get_msft(df, columns[i])
        r_t_i = get_rit(date1, date2, msft)

        rt += w_t_i * r_t_i

    return rt


date = 150
rt = get_indice(df, df2, t=date, N=300)


"""Exercise 3"""

"a"


def get_average_perf(year):
    all_date = df2["date"]
    dates = []
    for x in all_date:
        if str(x)[0:4] == year:
            dates.append(x)


get_average_perf(1)

