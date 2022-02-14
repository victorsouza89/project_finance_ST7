import pandas as pd
import yfinance as yf
import math

"""Exercise 1"""

path = "DataProjets.xlsx"

"""a)"""
df = pd.read_excel(path, sheet_name="Mapping")
msft = yf.Ticker(df["Tickers"][0])  # I don't know what to do with this


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


def get_indice(df2, t=100, N=300):
    rt = 0
    columns = df2.columns[1:]
    for i in range(N):
        # Calcul de r_t^i
        company = df2[columns[i]]
        r_i_t = 0
        if company[t - 1] == 0:
            r_i_t = 0
        else:
            r_i_t = company[t] / company[t - 1] - 1
        rt += r_i_t

    return rt


date = 150
rt = get_indice(df2, t=date, N=300)
