import numpy as np 
import pandas as pd 
import seaborn as sns
from statsmodels.tsa.stattools import kpss, adfuller
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
from statsmodels.tsa.arima.model import ARIMA
import quandl 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import statsmodels.api as sm

def check_num_stationarity(ts):
    diff = 0

    def check(a, confidence_interval=0.01):
        dftest = adfuller(a, autolag='AIC')
        kpsstest = kpss(a, regression='c')

        if dftest[1] < confidence_interval:   
            if kpsstest[1] > confidence_interval:   
                print('The time series data is stationary.')
                return 1
            else:
                print('The time series data is difference stationary.')
                return 0
        else:
            if kpsstest[1] > confidence_interval:
                print('The time series data is trend stationary.')
                return 1
            else:
                return 0 

    check_value = 0

    while check_value != 1:
        ts = (ts - ts.shift()).dropna()   
        diff += 1
        check_value = check(ts)

    print(f'Number of differencing needed is: {diff}')

def analyze_residuals(residuals):
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))

    ax[0, 0].plot(residuals[13:], label='Residuals')
    ax[0, 0].set_title('Time Series of Residuals')
    ax[0, 0].set_xlabel('Date')
    ax[0, 0].set_ylabel('Residual')
    ax[0, 0].legend()

    ax[0, 1].hist(residuals, color='blue', label='Histogram of Residuals')
    ax[0, 1].set_title('Histogram of Residuals')
    ax[0, 1].set_xlabel('Residual Value')
    ax[0, 1].set_ylabel('Frequency')
    ax[0, 1].legend()


    plot_acf(residuals, ax=ax[1, 0], lags=40, alpha=0.05, title='Autocorrelation of Residuals')


    ljung_box_result = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True, boxpierce=True)


    print("Ljung-Box Test Results:")
    display(ljung_box_result)

    plt.tight_layout()
    plt.show()

def evaluate_forecast(y_true, y_pred):
    epsilon = 0
    y_pred.index = y_true.index
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    metrics_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'MAPE (%)': [mape],
        'sMAPE (%)': [smape]
    })

    return metrics_df

def plot_line_graph(train_data, valid_data, predicted_valid_data):
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, color='blue', label='Train Data')
    plt.plot(valid_data, color='green', label='Valid Data')
    plt.plot(predicted_valid_data, color='red', label='Predicted Valid Data')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Train Data vs Valid Data vs Predicted Valid Data')
    plt.grid(False)
    plt.legend()
    plt.show()