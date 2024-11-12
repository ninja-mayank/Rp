import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import norm
import vae

df = pd.read_csv('/Users/mayanksood/Documents/Research Project/VAE/market_data/adj_close.csv')
df = df.drop(columns=['Date','MMM'])
col = df.columns
df_returns = df.pct_change().fillna(0)

def var_covar(returns):
    initial_portfolio = 100000
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    weights = np.full((497), 1/497)
    port_mean_return = (weights * mean_returns).sum()
    port_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
    confidence_level = 0.95
    z_score = norm.ppf(q=1-confidence_level)
    var = - (norm.ppf(confidence_level)*port_std_dev - port_mean_return)
    cvar = 1 * (port_mean_return - port_std_dev * (norm.pdf(z_score)/(1-confidence_level)))

    var_initial = initial_portfolio * var

    print(f"Parametric VaR at {confidence_level} confidence level: {var_initial:.2f} ({var:.2%})")

for i in range(1006 - 366):
    new_df = df_returns.iloc[i:i + 365].copy()
    gen = vae.synthetic_data(new_df)
    list = []

    net_loss = 0
    cnt = 0
    weights = np.full((497), 1/497)
    for g in gen:
        net_loss += weights[cnt] * g * 100000
        cnt += 1
    var_covar(new_df)
    print("Var from Variational Auto Encoders is",net_loss)
    loss = 0
    counter = 0
    for c in col:
        loss += weights[counter] * df_returns[c][i + 365] * 100000
        counter += 1

    print(loss)

    


    


