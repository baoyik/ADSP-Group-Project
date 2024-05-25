Part 1: Predicting stock price for the largest capped company in the industry
Tech Industry (with multiplicative trend): MSFT
Airline Industry (potentially with strong seasonality): DAL (Delta Airline)
Utility Industry (potentially with high autocorrelation): DUK 

Daily data fetching for the above three stocks from 2000-01-01 to 2024-01-01
Train-Valid-Test split: 0.6:0.2:0.2
1. Try the following models on the training set:  
Smoothing(on validation), ARIMA/SARIMA/Auto-ARIMA(subsetted data)/Prophet//VARMA(with QQQ or SPY)/TBATS/GARCH (Use BIC/AIC/Residual Analysis to determine one fitted model for each model type to proceed in the next step)
-for MSFT, compare it with QQQ(ETF that tracks Nasdaq index), QQQ could serve as an exogenous variable in SARIMA, try Granger Causality test in both direction

2.  Model selection on validation set: compaMre error metrics MAE/MSE/MAPE/sMAPE to          find one best model to proceed in the next step

3. Best model fit on test set: calculate error metrics, use MAE/MSE/MAPE/sMAPE to compare across three different stocks

Part 2: Sector Classification
20 Tech stock/20 Airline stock/20 Utility stock 
Airline: ['UAL','AAL','ALK','JBLU','DAL','LUV','SKYW','RYAAY','CPA',
                  'ALGT','SAVE','SNCY','BA','NCLH','VLRS','MESA','HA','AZUL',
                  'ATSG','GOL']
Utility: ['DUK','SO','NEE','D','SRE','EXC','CEG','PCG','AEP','GEV',
                  'XLU','RSPU','FUTY','IDU','VPU','PEG','XEL','NGG','WEC','AEE',
                  'ES','AWK','ETR','EIX','DTE','FE']
Tech: ['NVDA','AMD','AAPL','MSFT','CRM','CRWD','AVGO','INTC','ORCL','QCOM',
               'MU','INTU','AMAT','TTD','LRCX','NXPI','GOOG','ADBE','CSCO','TXN',
               'META','SNOW','DELL']
Data fetching four years of the 60 tickers above from 2021-03-17 to 2024-01-01
Train-Test split: 10 stock in training and 10 stock in test for each of the three sectors

1. Try machine learning models using hyperparameter tuning and cross-validation on the training set: Random forest/Gradient Boosting with different algorithms (Interval Based/Frequency Based/Shaplet Based)/SVM with tuned kernal
2. Compute the confusion matrix, recall rate and precision rate for each optimized model
3. Discuss the tradeoff between recall rate versus precision rate in final model selection

Task assignment
Part 1 MSFT prediction (1 person: Jiayu)
Part 1 DAL prediction (1 person: Alaia)
Part 1 DUK prediction (1 person: James)
Part 2 classification (2 person: Thea, Kelly )

Deadline
Thursday night - 9:30pm: zoom, every group member finish each part before the meeting
Friday: combines all the notebooks, create slides of presentation, divide part for presentation
Friday night - 8pm: zoom, presentation rehearsal

