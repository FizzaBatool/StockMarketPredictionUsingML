from datetime import datetime, date
import pandas as pd 
import numpy as np
import requests
import time
import math 
import pickle
# import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

stock=''
data = pd.DataFrame()
df = pd.DataFrame()

def EMAada(market):

    from psx import  tickers , stocks
        
    stock = market
    tickers = tickers()

    
    data = stocks(stock, start=date(2018, 1, 1), end=date.today())
    df= pd.DataFrame(data)
    
    x = df.index
    y = df['Close']

    df.reset_index(inplace=True) 
    df['Date'] = pd.to_datetime(df.Date)

    weights = np.arange(1,11) 
    wma10 = data['Close'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

    sma10 = data['Close'].rolling(10).mean()

    ema10 = data['Close'].ewm(span=10).mean()

    data['EMA'] = np.round(ema10, decimals=3)

    modPrice = data['Close'].copy()
    modPrice.iloc[0:10] = sma10[0:10]

    ema10alt = modPrice.ewm(span=10, adjust=False).mean()

    df['Close']=df['Close'].shift(-1)

    XNopen=df.iloc[-1]['Open']
    XNhigh= df.iloc[-1]['High']
    XNlow =df.iloc[-1]['Low']
    XNvolume= df.iloc[-1]['Volume']
    XNema= df.iloc[-1]['EMA']

    df.drop(index=df.index[-1],axis=0,inplace=True)

    df.fillna(0, inplace=True)

    x = df[['Open', 'High','Low', 'Volume','EMA']]
    y = df['Close']
    ada = AdaBoostRegressor(n_estimators=100, learning_rate=1, random_state = 1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,  random_state = 10)
    ada.fit(train_x, train_y)
    prediction = ada.predict(test_x)
    regression_confidence = ada.score(test_x, test_y)
   

    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNema]]
    ynew = ada.predict(xnew)

    return ynew

def RSIada(market):

    from psx import  tickers, stocks
        
    stock = market
    tickers = tickers()

    #pip install psx-data-reader
    
    data = stocks(stock, start=date(2018, 1, 1), end=date.today())
    df= pd.DataFrame(data)
    
    x = df.index
    y = df['Close']

    df.reset_index(inplace=True) 
    df['Date'] = pd.to_datetime(df.Date)

    def rsi(close, periods = 14):
          close_delta = df['Close'].diff()

          up = close_delta.clip(lower=0)
          down = -1 * close_delta.clip(upper=0)
          
          ma_up = up.ewm(com = 14 - 1, adjust=True, min_periods = 14).mean()
          ma_down = down.ewm(com = 14 - 1, adjust=True, min_periods = 14).mean()

          rsi = ma_up / ma_down
          rsi = 100 - (100/(1 + rsi))

          return rsi
    

    data['RSI'] = rsi(data['Close'])

    df['Close']=df['Close'].shift(-1)

    XNopen=df.iloc[-1]['Open']
    XNhigh= df.iloc[-1]['High']
    XNlow =df.iloc[-1]['Low']
    XNvolume= df.iloc[-1]['Volume']
    XNrsi= df.iloc[-1]['RSI']

    df.drop(index=df.index[-1],axis=0,inplace=True)
    #df.fillna(0, inplace=True)

    x = df[['Open', 'High','Low', 'Volume','RSI']]
    y = df['Close']

    df.fillna(0, inplace=True)
    # X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    ada = AdaBoostRegressor(n_estimators=100, learning_rate=1, random_state = 1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,  random_state = 10)
    ada.fit(train_x, train_y)
    prediction = ada.predict(test_x)
    regression_confidence = ada.score(test_x, test_y)

    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNrsi]]
    ynew = ada.predict(xnew)

    return ynew

def OBVada(market):

    from psx import  tickers , stocks
        
    stock = market
    tickers = tickers()

    
    data = stocks(stock, start=date(2018, 1, 1), end=date.today())
    df= pd.DataFrame(data)
    
    x = df.index
    y = df['Close']

    df.reset_index(inplace=True) 
    df['Date'] = pd.to_datetime(df.Date)


    weights = np.arange(1,11) 
    wma10 = data['Close'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

    sma10 = data['Close'].rolling(10).mean()

    ema10 = data['Close'].ewm(span=10).mean()

    data['OBV'] = np.round(ema10, decimals=3)

    modPrice = data['Close'].copy()
    modPrice.iloc[0:10] = sma10[0:10]

    ema10alt = modPrice.ewm(span=10, adjust=False).mean()

    df['Close']=df['Close'].shift(-1)

    XNopen=df.iloc[-1]['Open']
    XNhigh= df.iloc[-1]['High']
    XNlow =df.iloc[-1]['Low']
    XNvolume= df.iloc[-1]['Volume']
    XNobv= df.iloc[-1]['OBV']

    df.drop(index=df.index[-1],axis=0,inplace=True)

    df.fillna(0, inplace=True)

    x = df[['Open', 'High','Low', 'Volume','OBV']]
    y = df['Close']

    ada = AdaBoostRegressor(n_estimators=100, learning_rate=1, random_state = 1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,  random_state = 10)
    ada.fit(train_x, train_y)
    prediction = ada.predict(test_x)
    regression_confidence = ada.score(test_x, test_y)

    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNobv]]
    ynew = ada.predict(xnew)

    return ynew

def MFIada(market):

    from psx import  tickers , stocks
        
    stock = market
    tickers = tickers()

    #pip install psx-data-reader
    
    data = stocks(stock, start=date(2018, 1, 1), end=date.today())
    df= pd.DataFrame(data)
    
    x = df.index
    y = df['Close']

    df.reset_index(inplace=True) 
    df['Date'] = pd.to_datetime(df.Date)



    m = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = m - d
    df['MFI'] = df.index.map(macd)
    pd.set_option("display.max_columns", None)

    df['Close']=df['Close'].shift(-1)

   

    XNopen=df.iloc[-1]['Open']
    XNhigh= df.iloc[-1]['High']
    XNlow =df.iloc[-1]['Low']
    XNvolume= df.iloc[-1]['Volume']
    XNmfi= df.iloc[-1]['MFI']

    df.drop(index=df.index[-1],axis=0,inplace=True)

    df.fillna(0, inplace=True)

    x = df[['Open', 'High','Low', 'Volume','MFI']]
    y = df['Close']

    ada = AdaBoostRegressor(n_estimators=100, learning_rate=1, random_state = 1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,  random_state = 10)
    ada.fit(train_x, train_y)
    prediction = ada.predict(test_x)
    regression_confidence = ada.score(test_x, test_y)

    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNmfi]]
    ynew = ada.predict(xnew)

    return ynew

def MACDada(market):

    from psx import  tickers , stocks
        
    stock = market
    tickers = tickers()
    
    data = stocks(stock, start=date(2018, 1, 1), end=date.today())
    df= pd.DataFrame(data)
    
    x = df.index
    y = df['Close']

    df.reset_index(inplace=True) 
    df['Date'] = pd.to_datetime(df.Date)

    k = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = k - d
    df['MACD'] = df.index.map(macd)
    pd.set_option("display.max_columns", None)

    df['Close']=df['Close'].shift(-1)

    XNopen=df.iloc[-1]['Open']
    XNhigh= df.iloc[-1]['High']
    XNlow =df.iloc[-1]['Low']
    XNvolume= df.iloc[-1]['Volume']
    XNmacd= df.iloc[-1]['MACD']

    df.drop(index=df.index[-1],axis=0,inplace=True)
    df.fillna(0, inplace=True)

    x = df[['Open', 'High','Low', 'Volume','MACD']]
    y = df['Close']

    ada = AdaBoostRegressor(n_estimators=100, learning_rate=1, random_state = 1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,  random_state = 10)
    ada.fit(train_x, train_y)
    prediction = ada.predict(test_x)
    regression_confidence = ada.score(test_x, test_y)
    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNmacd]]
    ynew = ada.predict(xnew)

    return ynew
