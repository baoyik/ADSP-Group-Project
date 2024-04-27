import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import plotly.express as px
import pandas as pd
import numpy as np
import scipy as sp
import quandl
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import coint
from dataclasses import dataclass
from enum import Enum
from statsmodels.regression.rolling import RollingOLS



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

