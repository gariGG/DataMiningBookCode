import math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
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
    axes[0].set_ylabel('RIdership (in 000s)')
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








