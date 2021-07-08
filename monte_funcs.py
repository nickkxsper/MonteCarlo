import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance
import pylab
import random
import numpy
from IPython.display import clear_output


def pnl_walk(a, n, rates):
    # defining the number of steps

    #creating two array for containing x and y coordinate
    #of size equals to the number of size and filled up with 0's
    x = np.zeros(n)
    x[0] = a

    # filling the coordinates with random variables
    for i in range(1, n):
        rate = random.choice(rates)
        x[i] = x[i-1]*(1+rate)
    
    
    return x

def simulate_rndm_pnl(n_sims, starting_val, steps, rates):
    ends = []
    sers = []
    for i in range(n_sims):
        ser = pnl_walk(starting_val, steps, rates)
        ends.append(ser[-1])
        sers.append(ser)
    return sers, ends



def simulate_returns(data, rolling_lookback, n_paths, n_days_project, pred_col = 'Close'):
   

    cur = data[f'{pred_col}']
    rolling_sd = data['rolling_sd']
    rolling_mean = data['rolling_mean']
    generated = np.random.normal(rolling_mean,rolling_sd,1000)
    paths, ends = simulate_rndm_pnl(n_paths, cur, n_days_project,generated)
    df = pd.DataFrame(paths).transpose()
    #for col in df:
    #    plt.plot(df[col])
    #print(f'Current Price. {cur}')
    #print(f'Projected Mean in {n_days_project} days. {np.mean(ends)}')
    #print(f'Projected Standard Deviation in {n_days_project} days. {np.std(ends)}')
    return np.mean(ends), np.std(ends)


def historical_test(ticker, rolling_lookback, n_paths, n_days_project):
    data = yfinance.download(f'{ticker}')
    data['Pct_Change'] = data['Close'].pct_change()
    data['log_return'] = np.log(1 + data.Pct_Change)
    data['rolling_mean'] = data['Pct_Change'].rolling(rolling_lookback).mean()
    data['rolling_sd'] = data['Pct_Change'].rolling(rolling_lookback).std()
    data = data.dropna(axis = 0)
    E = []
    s = []
    for i in range(len(data) - n_days_project):
        print(str(int(i)/(len(data)-n_days_project)*100) + f'% done simulating {ticker}')
        clear_output(wait = True)
        sim_mu, sim_sd = simulate_returns(data.iloc[i], rolling_lookback, n_paths, n_days_project)
        E.append(sim_mu)
        s.append(sim_sd)
    data = data.iloc[0:len(data)-n_days_project]
    data['E'] = E
    data['s'] = s
    data['Actual'] = data['Close'].shift(-1*n_days_project)
    data['Error'] = data['Actual'] - data['E']
    return data