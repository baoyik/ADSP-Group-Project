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
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

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
    epsilon = 1e-8
    y_pred.index = y_true.index
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) 
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    metrics_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'MAPE': [mape],
        'sMAPE': [smape]
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


    display("Ljung-Box Test Results:")
    display(ljung_box_result)

    plt.tight_layout()
    plt.show()

def fit_smoothing(df,method='mul',period=12):
    
    fit = ExponentialSmoothing(df,
                            trend=method,
                            seasonal_periods=period, 
                            seasonal=method
                           ).fit()
    return fit


def evaluation(method_fit,y_test):
    # Fit the model, with selected alpha value
    fit = method_fit
    # Forecast over testing period
    y_pred = fit.forecast(len(y_test))
    y_pred.index = y_test.index
    
    # Calculate smape of test set performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test))
    smape = np.mean(2*np.abs(y_pred - y_test) / ((np.abs(y_test) + np.abs(y_pred))))
    return [mae,mse,mape,smape]

def tune_param(train,val,methods=['add','mul'],periods=[4,12,24,36]):
    result = []
    
    for method in methods:
        for period in periods:
            model = fit_smoothing(train,method,period)
            evaluation_i = evaluation(model,val)

            result_i = pd.Series([method,period]+evaluation_i)
            result.append(result_i)

    result = pd.concat(result,axis=1).T
    result.columns = ['methods','periods','mae','mse','mape','smape']

    return result


def error_metrics_prophet(model, train, test):
    # Prepare future dataframe for Prophet to make predictions on test set
    future = pd.DataFrame(test.ds)
    forecast = model.predict(future)
    
    # Extract forecasted values aligning with the test set
    y_pred = forecast['yhat'].values
    
    # Calculating error metrics
    mae = mean_absolute_error(test['y'], y_pred)
    mse = mean_squared_error(test['y'], y_pred)
    mape = mean_absolute_percentage_error(test['y'], y_pred)
    smape = np.mean(2 * np.abs(y_pred - test['y'].values) / (np.abs(test['y'].values) + np.abs(y_pred) + np.finfo(float).eps))

    # Naive forecast using the last observation from training set
    naive_forecast = train['y'].iloc[-1]
    naive_errors = np.abs(train['y'] - naive_forecast).mean()  # Mean absolute error of the naive forecast
    
    # MASE calculation
    mase = mae / naive_errors

    # Compiling metrics into a DataFrame
    df_metrics = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'MAPE': [mape],
        'sMAPE': [smape],
        'MASE': [mase]
    })

    return df_metrics