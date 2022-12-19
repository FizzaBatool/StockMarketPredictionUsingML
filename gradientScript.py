from datetime import datetime, date
import pandas as pd 
import numpy as np
import requests
import time
import math 

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import preprocessing


stock=''
data = pd.DataFrame()
df = pd.DataFrame()


def EMAgrad(market):

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
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,  shuffle=False, random_state=1)

    gredient = GradientBoostingRegressor(n_estimators=600, max_depth=5,learning_rate=0.01, min_samples_split=3)

    gredient = GradientBoostingRegressor()
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100,  random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
    gbr= gredient.fit(X_train, y_train)
    prediction = gbr.predict(X_test)
    xnew = [[XNopen,XNhigh,	XNlow, XNvolume, XNema]]
    ynew = gbr.predict(xnew)

    return ynew

def RSIgrad(market):

    from psx import  tickers, stocks
        
    stock = market
    tickers = tickers()
   
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
    df.fillna(0, inplace=True)

    x = df[['Open', 'High','Low', 'Volume','RSI']]
    y = df['Close']

  
    # X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,  shuffle=False, random_state=1)

    gredient = GradientBoostingRegressor(n_estimators=600, max_depth=5,learning_rate=0.01, min_samples_split=3)

    gredient = GradientBoostingRegressor()
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100,  random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
    gbr= gredient.fit(X_train, y_train)
    prediction = gbr.predict(X_test)
    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNrsi]]
    ynew = gbr.predict(xnew)

    return ynew

def OBVgrad(market):

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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,  shuffle=False, random_state=1)

    gredient = GradientBoostingRegressor(n_estimators=600, max_depth=5,learning_rate=0.01, min_samples_split=3)

    gredient = GradientBoostingRegressor()
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100,  random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
    gbr= gredient.fit(X_train, y_train)
    prediction = gbr.predict(X_test)
    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNobv]]
    ynew = gbr.predict(xnew)
  
    return ynew

def MFIgrad(market):

    from psx import  tickers , stocks
        
    stock = market
    tickers = tickers()

    data = stocks(stock, start=date(2018, 1, 1), end=date.today())
    df= pd.DataFrame(data)
    
    x = df.index
    y = df['Close']

    df.reset_index(inplace=True) 
    df['Date'] = pd.to_datetime(df.Date)

    m = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = m - d
    df['MACD'] = df.index.map(macd)
    pd.set_option("display.max_columns", None)

    df['Close']=df['Close'].shift(-1)

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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,  shuffle=False, random_state=1)

    gredient = GradientBoostingRegressor(n_estimators=600, max_depth=5,learning_rate=0.01, min_samples_split=3)

    gredient = GradientBoostingRegressor()
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100,  random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
    gbr= gredient.fit(X_train, y_train)
    prediction = gbr.predict(X_test)
    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNmfi]]
    ynew = gbr.predict(xnew)
    return ynew

def MACDgrad(market):

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
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,  shuffle=False, random_state=1)

    gredient = GradientBoostingRegressor(n_estimators=600, max_depth=5,learning_rate=0.01, min_samples_split=3)

    gredient = GradientBoostingRegressor()
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
              min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100,  random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
    gbr= gredient.fit(X_train, y_train)
    prediction = gbr.predict(X_test)
    xnew = [[XNopen,	XNhigh,	XNlow, XNvolume, XNmacd]]
    ynew = gbr.predict(xnew)
   
    return ynew

