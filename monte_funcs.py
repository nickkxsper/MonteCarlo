import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance
import pylab
import random
import numpy
from IPython.display import clear_output
import datetime
import math
from hurst import compute_Hc


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
    generated = np.random.lognormal(rolling_mean,rolling_sd,25000) -1
    paths, ends = simulate_rndm_pnl(n_paths, cur, n_days_project,generated)
    df = pd.DataFrame(paths).transpose()
    #for col in df:
    #    plt.plot(df[col])
    #print(f'Current Price. {cur}')
    #print(f'Projected Mean in {n_days_project} days. {np.mean(ends)}')
    #print(f'Projected Standard Deviation in {n_days_project} days. {np.std(ends)}')
    return np.mean(ends), np.std(ends)

def hurst_exp(series):
    return compute_Hc(series,kind = 'price', simplified = True)[0]

def historical_test(ticker, rolling_lookback, n_paths, n_days_project):
    data = yfinance.download(f'{ticker}')
    data['Pct_Change'] = data['Close'].pct_change()
    data['log_return'] = np.log(1 + data.Pct_Change)
    data['rolling_mean'] = data['Pct_Change'].rolling(rolling_lookback).mean()
    data['rolling_sd'] = data['Pct_Change'].rolling(rolling_lookback).std()
    #data['rolling_hurst'] = data['Close'].rolling(100).apply(hurst_exp)
    #print(data)
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
    data['Error'] = (data['Actual'] - data['E'])/data['Actual']
    
    return data


def sim_and_test(strat, rfr, starting_amt, max_draw, wait_after_stop, tkrs, rolling_lookback, n_paths, n_days_project):
    
    dfs = {}
    dtas = []
    for tkr in tkrs:#,'SPY', 'AAPL', 'FB', 'AMZN', 'BA', 'GM']:
        pnls = {}
        #print('eeee')
        dta = historical_test(ticker = tkr, rolling_lookback = rolling_lookback, n_paths = n_paths, n_days_project  = n_days_project)
        
        pnl = [starting_amt]
        pnl_tkr = [starting_amt]
        
        stopped_out = False
        stopped_out_before = False
        wait = 0
        trades = 0
        if strat == 'Long':
            for i in range(len(dta)-1):
                print(str(int(i)/(len(dta)-1)*100) + f'% done backtesting {tkr}')
                clear_output(wait = True)
                
                if not stopped_out_before:
                    max_val = max(pnl)
                else:
                    max_val = max(pnl[-1*stopped_val+1:])
                cur_val = pnl[i]
                if not stopped_out:
                    if (cur_val - max_val)/max_val < -1*max_draw:
                        stopped_out = True
                        stopped_out_before = True
                        stopped_val = i
                        print('stopped')

                if stopped_out and wait < wait_after_stop:
                    pnl.append(pnl[-1])
                    pnl_tkr.append(pnl_tkr[-1] * (1+dta.iloc[i+1]['Pct_Change']))
                    wait += 1
                    continue
                else:
                    wait = 0
                    stopped_out = False
                    pnl_tkr.append(pnl_tkr[-1] * (1+dta.iloc[i+1]['Pct_Change']))
                    if dta.iloc[i]['Close'] < dta.iloc[i]['E']:
                        trades +=1
                        pnl.append(pnl[-1]*(1+dta.iloc[i+1]['Pct_Change']))
                    else:
                        pnl.append(pnl[-1])
            
            
        if strat == 'Short':
            for i in range(len(dta)-1):
                print(str(int(i)/(len(dta)-1)*100) + f'% done backtesting {tkr}')
                clear_output(wait = True)
                
                if not stopped_out_before:
                    max_val = max(pnl)
                else:
                    max_val = max(pnl[-1*stopped_val+1:])
                cur_val = pnl[i]
                if not stopped_out:
                    if (cur_val - max_val)/max_val < -1*max_draw:
                        stopped_out = True
                        stopped_out_before = True
                        stopped_val = i
                        print('stopped')

                if stopped_out and wait < wait_after_stop:
                    pnl.append(pnl[-1])
                    pnl_tkr.append(pnl_tkr[-1] * (1+dta.iloc[i+1]['Pct_Change']))
                    wait += 1
                    continue

                else:
                    stopped_out = False
                    pnl_tkr.append(pnl_tkr[-1] * (1+dta.iloc[i+1]['Pct_Change']))
                    if dta.iloc[i]['Close'] > dta.iloc[i]['E']:
                        pnl.append(pnl[-1]*(1+(-1*dta.iloc[i+1]['Pct_Change'])))
                        trades +=1
                    else:
                        pnl.append(pnl[-1])

            
        if strat == 'LongShort':
            for i in range(len(dta)-1):
                print(str(int(i)/(len(dta)-1)*100) + f'% done backtesting {tkr}')
                clear_output(wait = True)
                
                max_val = max(pnl)
                cur_val = pnl[i]
                
                if not stopped_out_before:
                    max_val = max(pnl)
                else:
                    max_val = max(pnl[-1*stopped_val+1:])
                cur_val = pnl[i]
                if not stopped_out:
                    if (cur_val - max_val)/max_val < -1*max_draw:
                        stopped_out = True
                        stopped_out_before = True
                        stopped_val = i
                        print('stopped')

                if stopped_out and wait < wait_after_stop:
                    pnl.append(pnl[-1])
                    pnl_tkr.append(pnl_tkr[-1] * (1+dta.iloc[i+1]['Pct_Change']))
                    wait += 1
                    continue
                else:
                    pnl_tkr.append(pnl_tkr[-1] * (1+dta.iloc[i+1]['Pct_Change']))
                    if dta.iloc[i]['Close'] > dta.iloc[i]['E']:
                        trades +=1
                        pnl.append(pnl[-1]*(1+(-1*dta.iloc[i+1]['Pct_Change'])))
                    elif dta.iloc[i]['Close'] < dta.iloc[i]['E']:
                        trades +=1
                        pnl.append(pnl[-1]*(1+dta.iloc[i+1]['Pct_Change']))
                    else:
                        pnl.append(pnl[-1])
        print(f"Trades: {trades}")
      
        pnls[f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'] = pnl 
        pnls[f'Long_{tkr}'] = pnl_tkr

        df = pd.DataFrame(pnls)
        df['Strat_Ret'] = df[f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'] - df[f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'].shift(1)
        df = df.set_index(yfinance.download(f'{tkr}').index[:(-n_days_project -rolling_lookback)])
        dfs[f'{tkr}'] = df
        dtas.append(dta)
        #dfs.append(df)
    length = len(tkrs)
    keys = list(dfs.keys())
    if length > 1:
        fig, axs = plt.subplots(ncols = length,nrows = 2, squeeze=False, constrained_layout = True)
        for i in range(length):
            for col in dfs[keys[i]]:
                axs[i][0].plot(dfs[keys[i]][col], label = f'{col}')
            axs[i][0].set_xlabel('Date')
            axs[i][0].set_ylabel('Portfolio Value')
            axs[i][0].set_title(f'{tkrs[i]}_lookback={rolling_lookback}_n_paths={n_paths}_n_days_project={n_days_project}_backtest')
            
            axs[i][0].legend()
            axs[i][1].hist(dtas[i]['Error'])
            axs[i][1].set_xlabel('Error')
            axs[i][1].set_ylabel('Frequency')
            axs[i][1].set_title(f'{tkr}_prediction_errors')
    else:
        fig, axs = plt.subplots(2, squeeze=False, constrained_layout = True)
        for col in dfs[keys[0]]:
            axs[0][0].plot(dfs[keys[0]][col], label = f'{col}')
        axs[0][0].set_xlabel('Date')
        axs[0][0].set_ylabel('Portfolio Value')
        axs[0][0].set_title(f'{tkr}_lookback={rolling_lookback}_n_paths={n_paths}_n_days_project={n_days_project}_backtest')
        
        axs[0][0].legend()
        
        axs[1][0].hist(dta['Error'],bins = math.floor(np.sqrt(len(dta))))
        axs[1][0].set_xlabel('Error')
        axs[1][0].set_ylabel('Frequency')
        axs[1][0].set_title(f'{tkr}_prediction_errors')
        
        
    for tkr in tkrs:
        #mean_strat = np.mean(dfs[f'{tkr}'][f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'])
        num_years = ((datetime.datetime.strptime(str(dfs[f'{tkr}'].index[-1]),'%Y-%m-%d %H:%M:%S').year + (datetime.datetime.strptime(str(dfs[f'{tkr}'].index[-1]),'%Y-%m-%d %H:%M:%S').month)/12))  - ((datetime.datetime.strptime(str(dfs[f'{tkr}'].index[0]),'%Y-%m-%d %H:%M:%S').year + (datetime.datetime.strptime(str(dfs[f'{tkr}'].index[0]),'%Y-%m-%d %H:%M:%S').month)/12))  #2021-06-04 00:00:00
        
        total_ret_strat = (dfs[f'{tkr}'][f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'][-1] - dfs[f'{tkr}'][f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'][0])/dfs[f'{tkr}'][f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'][0]
        sd_tmp = dfs[f'{tkr}'].pct_change().replace([np.inf, -np.inf], np.nan).dropna(axis = 0)/100
        sd_strat = np.std(sd_tmp[f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'])
        #sd_strat = np.std(dfs[f'{tkr}'][f'{rolling_lookback}_{n_paths}_{n_days_project}_{tkr}'])
        sharpe_strat = (total_ret_strat- rfr*num_years)/sd_strat
        total_ret_strat *=100
        
        
        #mean_control = np.mean(dfs[f'{tkr}'][f'Long_{tkr}'])
        total_ret_hodl = (dfs[f'{tkr}'][f'Long_{tkr}'][-1] - dfs[f'{tkr}'][f'Long_{tkr}'][0])/dfs[f'{tkr}'][f'Long_{tkr}'][0]
        sd_tmp = dfs[f'{tkr}'].pct_change().replace([np.inf, -np.inf], np.nan).dropna(axis = 0)/100
        sd_control = np.std(sd_tmp[f'{tkr}'][f'Long_{tkr}'])
        sharpe_control = (total_ret_hodl - rfr*num_years)/sd_control
        total_ret_hodl *=100
        
        print(f'Strat Sharpe ({rolling_lookback}_{n_paths}_{n_days_project}_{tkr}): {sharpe_strat}, Buy and Hold Sharpe ({tkr}): {sharpe_control}')
        print(f'Return Strat(%): {total_ret_strat}, Return Buy and Hold(%): {total_ret_hodl}')
        
        print(f'Anualized Return Strat(%): {(1+ total_ret_strat/100)**(1/num_years)-1}, Anualized Return Buy and Hold(%): {(1+ total_ret_hodl/100)**(1/num_years)- 1}')
        print(f'Anualized Vol Strat(%): {(1+ sd_strat/100)**(1/num_years)-1}, Anualized Vol Buy and Hold(%): {(1+ sd_control/100)**(1/num_years)- 1}')
        print(f'Anualized Sharpe Strat: {sharpe_strat*np.sqrt(252)}, Anualized Sharpe Buy and Hold: {sharpe_control*np.sqrt(252)}')
        print('\n')

    return dfs