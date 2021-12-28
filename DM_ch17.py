import math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
from statsmodels.api import qqplot
import seaborn as sns
from statsmodels.tsa import tsatools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics import tsaplots


path = r'C:\Users\ggeor\OneDrive\PythonCodeProjects\DataMiningForBusinessAnalysis\DataMiningBook\dmba'
amtrak_df = pd.read_csv(path + r'\Amtrak.csv')
amtrak_df['Date'] = pd.to_datetime(amtrak_df.Month, format='%d/%m/%Y')
ridership_ts = pd.Series(amtrak_df.Ridership.values, index=amtrak_df.Date, name='ridership')
ridership_df = tsatools.add_trend(ridership_ts, trend='ct')
ridership_df.columns
ridership_lm = sm.ols(formula='ridership ~ trend', data=ridership_df).fit()

ax = ridership_ts.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
ax.set_ylim(1300, 2300)
ridership_lm.predict(ridership_df).plot(ax=ax)
ridership_lm.summary()
nValid = 36
nTrain = len(ridership_ts) - nValid
train_df = ridership_df[:nTrain]
valid_df = ridership_df[nTrain:]
train_df.columns
ridership_lm = sm.ols(formula='ridership ~ trend', data=train_df).fit()
ridership_lm.summary()
ax = train_df['ridership'].plot(color='C0', linewidth=0.75)
ridership_lm.predict(train_df).plot(ax=ax, color='orange')


def singleGraphLayout(ax, ylim, train_df, valid_df):
    ax.set_xlim('1990', '2004-6')
    ax.set_ylim(*ylim)
    ax.set_xlabel('Time')
    one_month = pd.Timedelta('31 days')
    xtrain = (min(train_df.index), max(train_df.index) - one_month)
    xvalid = (min(valid_df.index) + one_month, max(valid_df.index) - one_month)
    xtv = xtrain[1] + 0.5*(xvalid[0] - xtrain[1])

    ypos = 0.9 * ylim[1] + 0.1 * ylim[0]
    ax.add_line(plt.Line2D(xtrain, (ypos, ypos), color='black', linewidth=0.5))
    ax.add_line(plt.Line2D(xvalid, (ypos, ypos), color='black', linewidth=0.5))
    ax.axvline(x=xtv, ymin=0, ymax=1, color='black', linewidth=0.5)

    ypos = 0.925 * ylim[1] + 0.075 * ylim[0]
    ax.text('1995', ypos, 'Training')
    ax.text('2002-3', ypos, 'Validation')

# date: 25-Dec-2021
amtrak_df = pd.read_csv(path + r'\Amtrak.csv')
amtrak_df['Date'] = pd.to_datetime(amtrak_df.Month, format='%d/%m/%Y')
ridership_ts = pd.Series(amtrak_df.Ridership.values, index=amtrak_df.Date, name='Ridership')
ridership_df = tsatools.add_trend(ridership_ts, trend='ct')
ridership_df.head()
ridership_lm = sm.ols(formula='Ridership ~ trend', data=ridership_df).fit()
ridership_lm.summary()

# plot the same series
ax = ridership_ts.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
ax.set_ylim(1300, 2300)
ridership_lm.predict(ridership_df).plot(ax=ax)

nValid = 36
nTrain = len(ridership_ts) - nValid

train_df = ridership_df[:nTrain]
valid_df = ridership_df[nTrain:]

ridership_lm = sm.ols(formula='Ridership ~ trend', data=train_df).fit()
ridership_lm.summary()
ridership_lm.predict(train_df)
predict_df = ridership_lm.predict(valid_df)


def singleGraphLayout(ax, ylim, train_df, valid_df):
    ax.set_xlim('1990', '2004-6')
    ax.set_ylim(*ylim)
    ax.set_xlabel('Time')
    one_month = pd.Timedelta('31 days')
    xtrain = (min(train_df.index), max(train_df.index) - one_month)
    xvalid = (min(valid_df.index) + one_month, max(valid_df.index) - one_month)
    xtv = xtrain[1] + 0.5 * (xvalid[0] - xtrain[1])

    ypos = 0.9 * ylim[1] + 0.1 * ylim[0]
    ax.add_line(plt.Line2D(xtrain, (ypos, ypos), color='black', linewidth=0.5))
    ax.add_line(plt.Line2D(xvalid, (ypos, ypos), color='black', linewidth=0.5))
    ax.axvline(x=xtv, ymin=0, ymax=1, color='black', linewidth=0.5)

    ypos = 0.925 * ylim[1] + 0.075 * ylim[0]
    ax.text('1995', ypos, 'Training')
    ax.text('2002-3', ypos, 'Validation')


def graphLayout(axes, train_df, valid_df):
    singleGraphLayout(axes[0], [1300, 2550], train_df, valid_df)
    singleGraphLayout(axes[1], [-550, 550], train_df, valid_df)
    train_df.plot(y='Ridership', ax=axes[0], color='C0', linewidth=0.75)
    valid_df.plot(y='Ridership', ax=axes[0], color='C0', linewidth=0.75, linestyle='dashed')
    axes[1].axhline(y=0, xmin=0, xmax=1, color='black', linewidth=0.5)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Ridership (in 000s)')
    axes[1].set_ylabel('Forecast Errors')
    if axes[0].get_legend():
        axes[0].get_legend().remove()


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))
ridership_lm.predict(train_df).plot(ax=axes[0], color='C1')
ridership_lm.predict(valid_df).plot(ax=axes[0], color='C1', linestyle='dashed')
residual = train_df.Ridership - ridership_lm.predict(train_df)
residual.plot(ax=axes[1], color='C1')
residualValid = valid_df.Ridership - ridership_lm.predict(valid_df)
residualValid.plot(ax=axes[1], color='C1', linestyle='dashed')

graphLayout(axes, train_df, valid_df)

ridership_lm_linear = sm.ols(formula='Ridership ~ trend', data=train_df).fit()
predict_df_linear = ridership_lm_linear.predict(valid_df)

ridership_lm_expo = sm.ols(formula='np.log(Ridership) ~ trend', data=train_df).fit()
predict_df_expo = ridership_lm_expo.predict(valid_df)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3.75))
train_df.plot(y='Ridership', ax=ax, color='C0', linewidth=0.75, label='Training')
valid_df.plot(y='Ridership', ax=ax, color='C0', linewidth=0.75, linestyle='dashed', label='Validation')
singleGraphLayout(ax, [1300, 2600], train_df, valid_df)
ridership_lm_linear.predict(train_df).plot(color='C1')
ridership_lm_linear.predict(valid_df).plot(color='C1', linestyle='dashed')
ridership_lm_expo.predict(train_df).apply(lambda row: math.exp(row)).plot(color='C2')
ridership_lm_expo.predict(valid_df).apply(lambda row: math.exp(row)).plot(color='C2', linestyle='dashed')
ax.get_legend().remove()

ridership_lm_poly = sm.ols(formula='Ridership ~ trend + np.square(trend)', data=train_df).fit()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))
ridership_lm_poly.predict(train_df).plot(ax=axes[0], color='C1')
ridership_lm_poly.predict(valid_df).plot(ax=axes[0], color='C1', linestyle='--')
residualPoly = train_df.Ridership - ridership_lm_poly.predict(train_df)
residualPoly.plot(ax=axes[1], color='C1')
residualPolyValid = valid_df.Ridership - ridership_lm_poly.predict(valid_df)
residualPolyValid.plot(ax=axes[1], color='C1', linestyle='dashed')
graphLayout(axes=axes, train_df=train_df, valid_df=valid_df)
ridership_lm_poly.summary()
ridership_lm_expo.summary()


ridership_df = tsatools.add_trend(ridership_ts, trend='c')
ridership_df['Month'] = ridership_df.index.month
ridership_df.head()
train_df = ridership_df[:nTrain]
valid_df = ridership_df[nTrain:]
ridership_lm_seasonal = sm.ols(formula='Ridership ~ C(Month)', data=train_df).fit()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))
ridership_lm_seasonal.predict(train_df).plot(ax=axes[0], color='C1')
ridership_lm_seasonal.predict(valid_df).plot(ax=axes[0], color='C1', linestyle='dashed')
resid_seasonal_train = train_df.Ridership - ridership_lm_seasonal.predict(train_df)
resid_seasonal_valid = valid_df.Ridership - ridership_lm_seasonal.predict(valid_df)
resid_seasonal_train.plot(ax=axes[1], color='C1')
resid_seasonal_valid.plot(ax=axes[1], color='C1', linestyle='dashed')
graphLayout(axes=axes, train_df=train_df, valid_df=valid_df)

# trend and seasonal model
ridership_df = tsatools.add_trend(ridership_ts, trend='ctt')
ridership_df['Month'] = ridership_df.index.month
train_df = ridership_df[:nTrain]
valid_df = ridership_df[nTrain:]
formula = 'Ridership ~ trend + trend_squared + C(Month)'
ridership_lm_trend_seasonal = sm.ols(formula=formula, data=train_df).fit()
ridership_lm_trend_seasonal.summary()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))
ridership_lm_trend_seasonal.predict(train_df).plot(ax=axes[0], color='C1')
ridership_lm_trend_seasonal.predict(valid_df).plot(ax=axes[0], color='C1', linestyle='dashed')
resid_trend_season_train = train_df.Ridership - ridership_lm_trend_seasonal.predict(train_df)
resid_trend_season_valid = valid_df.Ridership - ridership_lm_trend_seasonal.predict(valid_df)
resid_trend_season_train.plot(ax=axes[1], color='C2')
resid_trend_season_valid.plot(ax=axes[1], color='C2', linestyle='--')
graphLayout(axes=axes, train_df=train_df, valid_df=valid_df)
tsaplots.plot_acf(train_df['1991-01-01':'1993-01-01'].Ridership)
tsaplots.plot_acf(resid_trend_season_train, lags=12)

# test for github
# write new row to commit to the github
amtrak_df.columns
ridership_ts = pd.Series(amtrak_df.Ridership.values, index=amtrak_df.Date, name='Ridership')
ridership_df = tsatools.add_trend(ridership_ts, trend='ctt')
ridership_df['month'] = ridership_df.index.month
nValid = 36
nTrain = len(ridership_ts) - nValid
train_df = ridership_df[:nTrain]
valid_df = ridership_df[nTrain:]
formula = 'Ridership ~ trend + trend_squared + C(month)'
lmod_trend_seasonal = sm.ols(formula=formula, data=train_df).fit()
resid_trend_seasonal = lmod_trend_seasonal.resid

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8.5))
tsaplots.plot_acf(ridership_ts, ax=axes[0][0], lags=12)
tsaplots.plot_pacf(ridership_ts, ax=axes[0][1], lags=12)
tsaplots.plot_acf(resid_trend_seasonal, lags=12, ax=axes[1][0], title='ACF on Resid of LR')
tsaplots.plot_pacf(resid_trend_seasonal, lags=12, ax=axes[1][1], title='PACF on resid of LR')
train_resid_arima = ARIMA(resid_trend_seasonal, order=(1, 0, 0), freq='MS').fit(trend='nc', disp=0)
forecast, _, conf_int = train_resid_arima.forecast(1)
print(pd.DataFrame({'coef': train_resid_arima.params, 'std err': train_resid_arima.bse}))
print('Forecast {0: .3f} [{1[0][0]: .3f}, {1[0][1]: .3f}]'.format(forecast, conf_int))
conf_int[0][1]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 7.5))
tsaplots.plot_acf(train_resid_arima.resid, ax=axes[0], lags=12)
tsaplots.plot_pacf(train_resid_arima.resid, ax=axes[1], lags=12)

ax = lmod_trend_seasonal.resid.plot(figsize=(9, 4))
train_resid_arima.fittedvalues.plot(ax=ax)
singleGraphLayout(ax=ax, ylim=[-250, 250], train_df=train_df, valid_df=valid_df)


sp500_df = pd.read_csv(path + r'\SP500.csv')
sp500_df.head()
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'], format='%d-%b-%y').dt.to_period('M')
sp500_df.head()
sp500_ts = pd.Series(sp500_df.Close.values, index=sp500_df.Date, name='sp500')
sp500_ts.plot()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))
tsaplots.plot_acf(sp500_ts, lags=12, ax=axes[0])
axes[0].set_title('ACF of SP 500 Close price')
tsaplots.plot_pacf(sp500_ts, lags=12, ax=axes[1], title='PACF of SP 500 Close price')
sp500_arima = ARIMA(sp500_ts, order=(1, 0, 0)).fit(disp=0)
print(pd.DataFrame({'coef': sp500_arima.params, 'std err': sp500_arima.bse}))

# problem 17.1
air_df = pd.read_csv(path + r'\Sept11Travel.csv')
air_df.head()
air_df['Month'] = pd.to_datetime(air_df.Month, format='%b-%y')
air_df.columns
air_ts = pd.Series(air_df['Air RPM (000s)'].values, index=air_df.Month, name='air')
air_ts = [float(i.replace(',', '')) for i in air_ts.values]
air_ts
air_ts = pd.Series(air_ts, index=air_df.Month, name='air')
air_ts = air_ts/1000
air_ts.head()
train_air = air_ts[:'2001-09']
valid_air = air_ts['2001-09':]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 10))
train_air.plot(ax=axes[0], linewidth=0.75)
valid_air.plot(ax=axes[1], linewidth=0.75)

air_df = tsatools.add_trend(air_ts, trend='ct')
air_df['month'] = air_df.index.month
train_air = air_df[:'2001-09']
valid_air = air_df['2001-09':]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.5, 5.5))
tsaplots.plot_acf(train_air['air'], lags=12, ax=axes[0])
tsaplots.plot_pacf(train_air['air'], lags=12, ax=axes[1])
lmod_trend_seasonal_air = sm.ols(formula='air ~ trend + month', data=train_air).fit()
lmod_trend_seasonal_air.summary()
lmod_trend_seasonal_air_resid = lmod_trend_seasonal_air.resid

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.5, 5.5))
qqplot(data=lmod_trend_seasonal_air_resid, line='q', ax=axes[0])
axes[0].set_title('QQ Plot of LM residual')
tsaplots.plot_acf(lmod_trend_seasonal_air_resid, lags=12, ax=axes[1], title='ACF of residual of LM')
tsaplots.plot_pacf(lmod_trend_seasonal_air_resid, lags=12, ax=axes[2], title='PACF of residual of LM')

ar_air = ARIMA(lmod_trend_seasonal_air_resid, order=(1, 0, 0)).fit(trend='nc')
ar_air.summary()
ar_air_train_resid = ar_air.resid

plt.style.use('ggplot')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5.5))
qqplot(data=ar_air_train_resid, line='q', ax=axes[0])
axes[0].set_title('QQ plot of AR(1)')
tsaplots.plot_acf(ar_air_train_resid, lags=12, ax=axes[1], title='ACF of AR(1)')
axes[1].set_title('ACF on residual of AR(1)')
tsaplots.plot_pacf(ar_air_train_resid, lags=12, ax=axes[2], title='PACF on residual of AR(1)')

# date: 27-Dec-2021
air_df = pd.read_csv(path + r'\Sept11Travel.csv')
air_df.head()
air_df['Month'] = pd.to_datetime(air_df['Month'], format='%b-%y')
air_df.columns = [i.replace(' ', '_') for i in air_df.columns]
air_df.tail()
air_ts = pd.Series([float(i.replace(',', '')) for i in air_df['Air_RPM_(000s)'].values],
                   index=air_df.Month, name='air')
fig, ax = plt.subplots()
air_ts.plot(ax=ax)

air_an = tsatools.add_trend(air_ts, trend='ct')
air_an['month'] = air_an.index.month
air_an.air = air_an.air/1000000
train_air = air_an[:'08-2001']
valid_air = air_an['2001-08':]
train_air.columns
lmod_trend_season_train = sm.ols(formula='air ~ trend + C(month)', data=train_air).fit()
lmod_trend_season_train.summary()
grid = plt.GridSpec(nrows=2, ncols=2)
ax1 = plt.subplot(grid[0, :])
train_air.air.plot(ax=ax1, color='C1', linewidth=0.75)
ax1.set_title('Raw Data - Actual airline revenue passenger miles (Air)')
ax2 = plt.subplot(grid[1, 0])
train_air_resid = lmod_trend_season_train.resid
tsaplots.plot_acf(train_air_resid, ax=ax2, lags=12, title='ACF on LM residual')
ax3 = plt.subplot(grid[1, 1])
tsaplots.plot_pacf(train_air_resid, ax=ax3, lags=12, title='PACF on LM residual')
plt.tight_layout()

grid = plt.GridSpec(nrows=2, ncols=2)
ax1 = plt.subplot(grid[0, :])
train_air_resid.plot(ax=ax1, linewidth=0.75, color='C2')
ax1.axhline(0, linestyle='dashed', linewidth=0.9, color='black')
ax1.set_title('Residual on Linear-Regression model, train sample of AIR')
ax2 = plt.subplot(grid[1, 0])
tsaplots.plot_acf(train_air_resid, lags=12, ax=ax2, title='ACF on LM Residual')
ax3 = plt.subplot(grid[1, 1])
tsaplots.plot_pacf(train_air_resid, lags=12, ax=ax3, title='PACF on LM Residual')

fig, ax = plt.subplots()
lmod_trend_season_train.fittedvalues.plot(ax=ax, color='C1')

fig, axes = plt.subplots(nrows=2, ncols=1)
ax = lmod_trend_season_train.predict(train_air).plot(ax=axes[0], color='C1', linewidth=0.75, label='training')
lmod_trend_season_train.predict(valid_air).plot(ax=ax, color='C1', linewidth=0.75,
                                                linestyle='dashed', label='validation')
train_air.air.plot(ax=axes[0], color='C0', linewidth=1.0)
valid_air.air.plot(ax=axes[0], color='C0', linestyle='dashed', linewidth=1)

resid_train = train_air.air - lmod_trend_season_train.predict(train_air)
resid_train.plot(ax=axes[1], color='C1', linewidth=0.75)
resid_valid = valid_air.air - lmod_trend_season_train.predict(valid_air)
resid_valid.plot(ax=axes[1], color='C1', linewidth=0.75, linestyle='--')

train_air_ar = ARIMA(train_air_resid, order=(1, 0, 0)).fit(trend='nc')
print(pd.DataFrame({'coef': train_air_ar.params, 'std dev': train_air_ar.bse}))
train_air_resid_ar = train_air_ar.resid


def chartAnalysis(data, ax_title=[]):
    grid = plt.GridSpec(nrows=2, ncols=2)
    ax1 = plt.subplot(grid[0, :])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[1, 1])

    data.plot(ax=ax1, color='C1', linewidth=0.75)
    ax1.set_title(ax_title[0])
    tsaplots.plot_acf(data, lags=12, ax=ax2, title=ax_title[1])
    tsaplots.plot_pacf(data, lags=12, ax=ax3, title=ax_title[2])

    return None


chartAnalysis(data=train_air_resid_ar, ax_title=['AR(1) residual of the AIR',
                                                 'ACF of AR(1) residual', 'PACF of AR(1) residual'])
air_df.columns = [i.replace(' ', '_') for i in air_df.columns]
air_df.columns
rail = pd.Series([float(i.replace(',', '')) for i in air_df['Rail_PM'].values], index=air_df['Month'],
                 name='Rail')
fig, ax = plt.subplots()
rail.plot(ax=ax, color='C0', linewidth=0.75)
rail_df = tsatools.add_trend(rail, trend='ctt')
rail_df['month'] = rail.index.month
rail_df.columns
lmod_seasonal_rail = sm.ols(formula='Rail ~ C(month)', data=rail_df).fit()
resid_rail_seasonal = lmod_seasonal_rail.resid
fig, axes = plt.subplots(nrows=2, ncols=1)
rail_df.Rail.plot(ax=axes[0], color='C1', linewidth=0.75)
resid_rail_seasonal.plot(ax=axes[1], color='C1', linewidth=0.75)

lmod_trend_seasonal_rail = sm.ols(formula='Rail ~ trend + C(month)', data=rail_df).fit()
resid_trend_seasonal_rail = lmod_trend_seasonal_rail.resid


ax = lmod_trend_seasonal_rail.predict(rail_df).plot(ax=ax, color='C1', linewidth=0.75)
resid_trend_seasonal_rail = lmod_trend_seasonal_rail.resid
pred_trend_seasonal_rail = lmod_trend_seasonal_rail.predict(rail_df)
chartAnalysis(data=pred_trend_seasonal_rail, ax_title=['Predict of LR model: trend plus seasonal',
                                                       'ACF on residual of LM', 'PACF of residual on LM'])
rail_df.Rail.plot(ax=ax, color='C0', linewidth=1)

rail_trend_only_ar = ARIMA(resid_trend_seasonal_rail, order=(1, 0, 0)).fit()
print(pd.DataFrame({'coef': rail_trend_only_ar.params, 'std err': rail_trend_only_ar.bse}))

rail_df.columns
lmod_trend_seasonal_rail2 = sm.ols(formula='Rail ~ trend + trend_squared + C(month)', data=rail_df).fit()
lmod_trend_seasonal_rail2.summary()
resid_trend_seasonal_rail2 = lmod_trend_seasonal_rail2.resid
rail_trend_ar = ARIMA(resid_trend_seasonal_rail2, order=(1, 0, 0)).fit()
print(pd.DataFrame({'coef': rail_trend_ar.params, 'std err': rail_trend_ar.bse}))


fig, axes = plt.subplots(nrows=1, ncols=2)
tsaplots.plot_acf(rail_trend_only_ar.resid, lags=12, ax=axes[0], title='ACF of resid on LM trend only')
tsaplots.plot_acf(rail_trend_ar.resid, lags=12, ax=axes[1], title='ACF of resid on LM trend 2')

# date: 28-Dec-2021
# problem: 17.2

workHours = pd.read_csv(path + r'\CanadianWorkHours.csv')
workHours.head()
workHours['Year'] = pd.to_datetime(workHours['Year'], format='%Y')
workHours_ts = pd.Series(workHours['Hours'].values, index=workHours['Year'], name='hours')
fig, ax = plt.subplots()
workHours_ts.plot(ax=ax, color='green', linewidth=0.75)

grid = plt.GridSpec(nrows=2, ncols=2)
ax1 = plt.subplot(grid[0, :])
workHours_ts.plot(ax=ax1, color='green', linewidth=0.75)
ax1.set_title('Average canadian working hours per year')
ax2 = plt.subplot(grid[1, 0])
tsaplots.plot_acf(workHours_ts, ax=ax2, lags=12, title='ACF of Raw Data')
ax3 = plt.subplot(grid[1, 1])
tsaplots.plot_pacf(workHours_ts, lags=12, title='PACF of Raw Data', ax=ax3)

workHours_df = tsatools.add_trend(workHours_ts, trend='ctt')
workHours_df.head()
lmod_workHour = sm.ols(formula='hours ~ trend + trend_squared', data=workHours_df).fit()
lmod_workHour.summary()

fig, axes = plt.subplots(nrows=2, ncols=1)
lmod_workHour.predict(workHours_df).plot(ax=axes[0], color='C1', linewidth=0.75)
resid_trend2 = workHours_df.hours - lmod_workHour.predict(workHours_df)
workHours_df.hours.plot(ax=axes[0], color='green', linewidth=0.75)
axes[0].set_title('Linear Regression - Trend only')
resid_trend2.plot(ax=axes[1], color='C1', linewidth=0.75)
plt.tight_layout()

trend_ar = ARIMA(resid_trend2, order=(1, 0, 0)).fit()
print(pd.DataFrame({'coef': trend_ar.params, 'std err': trend_ar.bse}))


























