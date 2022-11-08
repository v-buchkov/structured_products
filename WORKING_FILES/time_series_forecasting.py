import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Use grid search to find the best parameters algorithmically

asset = web.get_data_yahoo(['AAPL'], start=dt.datetime(2018, 1, 1), end=dt.datetime(2022, 12, 2))['Close']
print(asset.head())
asset.to_csv('Asset_Data.csv')

asset = pd.read_csv("Asset_Data.csv")
asset.index = pd.to_datetime(asset['Date'], format='%Y-%m-%d')

del asset['Date']
# sns.set()
#
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
#
# plt.plot(asset.index, asset['AAPL'], )

train = asset[asset.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test = asset[asset.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]

# plt.plot(train, color="black")
# plt.plot(test, color="red")
# plt.ylabel('Asset')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.title("Train/Test split for AAPl Data")
# plt.show()

timeseries = train['AAPL']

"""
ARMA
"""
# Fitting
ARMA_model = SARIMAX(timeseries, order=(1, 0, 1))
ARMA_model = ARMA_model.fit()

# Generating predictions
y_pred = ARMA_model.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha=0.05)
y_pred_df["Predictions"] = ARMA_model.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index
# y_pred_out = y_pred_df["Predictions"]

# Calculating mean-squared error (the lower, the better)
ARMA_rmse = np.sqrt(mean_squared_error(test["AAPL"].values, y_pred_df["Predictions"]))
print("RMSE: ", ARMA_rmse)

"""
ARIMA(AR lag, differencing, MA lag)
"""
ARIMAmodel = ARIMA(timeseries, order=(2, 3, 2))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha=0.05)
y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index
# y_pred_out = y_pred_df["Predictions"]
# plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
# plt.legend()
ARIMA_rmse = np.sqrt(mean_squared_error(test["AAPL"].values, y_pred_df["Predictions"]))
print("RMSE: ", ARIMA_rmse)

"""
ARIMA(AR lag, differencing, MA lag)
"""
SARIMAXmodel = SARIMAX(timeseries, order=(5, 4, 2), seasonal_order=(2, 2, 2, 12))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha=0.05)
y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index
# y_pred_out = y_pred_df["Predictions"]
# plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
# plt.legend()

SARIMA_rmse = np.sqrt(mean_squared_error(test["AAPL"].values, y_pred_df["Predictions"]))
print("RMSE: ", ARMA_rmse)
