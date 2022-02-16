import pandas as pd
import yfinance as yf
import math

"""Exercise 1"""

path = "DataProjets.xlsx"

"""a)"""
df = pd.read_excel(path, sheet_name="Mapping")
msft = yf.Ticker(df["Tickers"][0])  # I don't know what to do with this


todays_data = msft.history(period='1d')

print(todays_data['Close'][0])

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

print(get_price("2021-02-01",df["Tickers"][0]))
