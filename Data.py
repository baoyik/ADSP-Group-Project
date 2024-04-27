import numpy as np 
import pandas as pd 
import seaborn as sns
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sktime.forecasting.arima import AutoARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from sktime.forecasting.base import ForecastingHorizon
import warnings
warnings.filterwarnings('ignore')

import quandl 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import statsmodels.api as sm


myAPIkey = 'VPTMgg7k44QB9_2PCKWD'
quandl.ApiConfig.api_key = myAPIkey

def get_ETF_data(name, start, end): 
    raw_data = quandl.get_table('QUOTEMEDIA/PRICES', ticker = name, date = {'gte':start, 'lte':end}) 
    # arrange the df in time series 
    df = raw_data.sort_values(by = 'date')
    # set time as index
    df.set_index('date', inplace = True)
    df  =  df.loc[:,['adj_close']]
    return df

def split_time_series(data, train_ratio, val_ratio):
    total_length = len(data)
    train_length = int(total_length * train_ratio)
    val_length = int(total_length * val_ratio)
    test_length = total_length - train_length - val_length
    
    train_data = data[:train_length]
    val_data = data[train_length:train_length+val_length]
    test_data = data[train_length+val_length:]
    
    return train_data, val_data, test_data


def check_data_summary(data):

    num_rows, num_cols = data.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")

    print("Column names:")
    print(data.columns)

    print("Data types:")
    print(data.dtypes)

    print("Summary statistics:")
    print(data.describe())

    print("Missing values:")
    print(data.isnull().sum())

    duplicated_index = data.index[data.index.duplicated()]
    if duplicated_index.empty:
        print("No duplicated index values.")
    else:
        print("Duplicated index values:")
        print(duplicated_index)
        index_format = data.index[0].strftime("%Y-%m-%d")
        for index in duplicated_index:
            if index.strftime("%Y-%m-%d") != index_format:
                print("Duplicated index values do not follow the same format.")
                break
        else:
            print("Duplicated index values follow the same format.")

def plot_stock_price(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values)  
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Over Time')
    plt.grid(False)  # Remove the grid
    plt.tight_layout()  # Adjust the layout
    plt.show()

def decompose_and_plot(data, freq):

    decomposition = seasonal_decompose(data, model='additive', period=freq)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(data, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def check_stationarity(df, trend='c', confidence_interval = 0.05, more_info = True):
    for c in df.columns: 
        time_series = df[c].dropna()
        # utilized the code from stationarity 
        dftest = adfuller(time_series, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        if more_info == True: # a detailed 
            print ('Results of Dickey-Fuller Test for '+ c +':')
            print (dfoutput)

        kpsstest = kpss(time_series, regression=trend)
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
        if more_info == True:
            print ('Results of KPSS Test for '+ c +':')
            print (kpss_output)

        #criteria: slide 38 lec 1
        if dftest[1] < confidence_interval: 
            if kpsstest[1] > confidence_interval:
                print(c + ' time series data is stationary')
            elif kpsstest[1] <= confidence_interval:
                print(c + ' time series data is difference stationary')
        else: 
            if kpsstest[1] > confidence_interval:
                print(c + ' time series data is trend stationary')
            elif kpsstest[1] <= confidence_interval:
                print(c + ' time series data is not stationary')


def plot_time_series_analysis(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataframe.")
    
    plt.figure(figsize=(6, 3))
    plot_acf(df[column], lags=20, alpha=0.05)   
    plt.title('Autocorrelation Function')
    plt.show()

    plt.figure(figsize=(6, 3))
    plot_pacf(df[column], lags=20, alpha=0.05, method='ywm')   
    plt.title('Partial Autocorrelation Function')
    plt.show()
