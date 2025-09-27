import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import *

np.random.seed(123)

n_list = [200, 500, 1000, 5000, 10000]
N_list = [1,2,5,10]
alpha_list=np.linspace(0.5, 5, 10)
beta_list=np.linspace(0.5, 5, 10)
sigma_2=1

rows=[]

for n in n_list:
    for alpha in alpha_list:
        for beta in beta_list:
            N_opt=compute_N_optimal(n,alpha,beta,sigma_2,N_list)
            h_AMISE, theta22_hat, sigma2_hat, RSS, support_length = optimal_bandwidth(n,alpha,beta,sigma_2,N_opt)
            rows.append({"bandwidth": h_AMISE, 
                        "dim_sample": n,
                        "num_blocks_opt": N_opt,
                        "alpha": alpha,
                        "beta": beta,
                        "residual_sum_of_square": RSS,
                        "support_length": support_length})

df_using_N_opt=pd.DataFrame(rows)

rows2=[]

for n in n_list:
    for alpha in alpha_list:
        for beta in beta_list:
            for N in N_list:
                h_AMISE, theta22_hat, sigma2_hat, RSS, support_length = optimal_bandwidth(n,alpha,beta,sigma_2,N)
                rows2.append({"bandwidth": h_AMISE, 
                            "dim_sample": n,
                            "num_blocks": N,
                            "alpha": alpha,
                            "beta": beta,
                            "residual_sum_of_square": RSS,
                            "support_length": support_length})
                
df_using_various_N=pd.DataFrame(rows2)





