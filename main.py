import pandas as pd
import yfinance as yf
import math

"""Exercise 1"""

path = "DataProjets.xlsx"

"""a)"""
df = pd.read_excel(path, sheet_name="Mapping")


def get_msft(df, sedol):
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


def convert_date(date):
    return str(date.year) + "-" + str(date.month) + "-" + str(date.day)


def get_price(msft, date):
    todays_data = msft.history(period="20y")["Close"]

    return todays_data["Close"][date]


def get_rit(df, date, sebol):
    msft = get_msft(df, sebol)
    P_i_t = get_price(msft, convert_date(date))
    P_i_tbefore = get_price(msft, convert_date(date))
    if (P_i_tbefore) == 0:
        return -1
    else:
        return P_i_t


date = df2["date"][226]
date_convert = convert_date(date)

sebol = df2.columns[1]
msft = get_msft(df, sebol)

todays_data = msft.history(period="20y")
close = todays_data["Close"]

get_rit(df, date=date, sebol=sebol)


def get_indice(df2, t=100, N=300):
    rt = 0
    columns = df2.columns[1:]
    for i in range(N):
        company = df2[columns[i]]

        w_t_i = df2[i]

        r_t_i = 0
        r_t_i = get_rit(df, df2["date"][t])

        rt += w_t_i * r_t_i

    return rt


date = 150
rt = get_indice(df2, t=date, N=300)
