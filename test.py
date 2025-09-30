import numpy as np
from helpers import *
import seaborn as sns
import matplotlib.pyplot as plt

#example with X_i~Beta(2,2), number of sample n=1000, number of blocks N=6

np.random.seed(1)
alpha, beta=2,2
n=500
N_list=[1,2,5,10]
sigma_2=0.3

m= lambda x: np.sin((x/3+0.1)**(-1))

N=compute_N_optimal(n,alpha,beta,sigma_2,N_list)
h, X, Y, *_=optimal_bandwidth(n,alpha,beta,sigma_2,N)
x_eval=np.linspace(np.min(X), np.max(X), 5000)

K= lambda x: ((15/16)*(1-x**2)**2)*(np.abs(x)<=1) #quartic kernel
S= lambda k,x: (1/(n*h))*np.sum(K((X-x)/h)*(X-x)**k)
w= lambda x:(1/(n*h))*(K((X-x)/h)*(S(2,x)-(X-x)*S(1,x)))/(S(0,x)*S(2,x)-S(1,x)**2)

m_hat = np.array([np.sum(w(x) * Y) for x in x_eval])


sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 8))

plt.scatter(
    X, Y, 
    s=12, color="dodgerblue", alpha=0.6, edgecolor="k", linewidth=0.3,
    label="Sample points"
)

plt.plot(
    x_eval, m(x_eval), 
    color="crimson", linewidth=2.5, linestyle="--",
    label=r"True regression $m(x)$"
)

plt.plot(
    x_eval, m_hat, 
    color="seagreen", linewidth=2.5,
    label=r"Estimation $\hat{m}(x)$"
)

plt.title(
    "Nonparametric Regression with Local Polynomial Estimator",
    fontsize=18, fontweight="bold", pad=20
)
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)

plt.legend(
    fontsize=12, frameon=True, fancybox=True, shadow=True,
    loc="upper right", borderpad=1
)

plt.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

sns.despine()

plt.show()
