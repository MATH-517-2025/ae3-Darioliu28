import numpy as np

def data_generation(n, alpha, beta, sigma_2):
    epsilon=np.random.normal(loc=0, scale=sigma_2, size=n)
    X=np.random.beta(a=alpha, b=beta, size=n)

    m= lambda x: np.sin((x/3+0.1)**(-1))
    Y=m(X)+epsilon

    return X,Y

def blocks_creator(X, Y, N):
    quantiles = np.quantile(X, np.linspace(0,1,N+1))
    blocks_X, blocks_Y = [], []
    
    for j in range(N):
        if j < N-1:
            mask = (X >= quantiles[j]) & (X < quantiles[j+1])
        else: 
            mask = (X >= quantiles[j]) & (X <= quantiles[j+1])
        blocks_X.append(X[mask])
        blocks_Y.append(Y[mask])
    return blocks_X, blocks_Y

def compute_diff_2_m(beta,Xi):
    return 2*beta[2] + 6*beta[3]*Xi + 12*beta[4]*Xi**2

def ols_etimate(Xi,Yi):
    X=np.vander(Xi, N=5, increasing=True)
    beta_hat, *_ = np.linalg.lstsq(X, Yi, rcond=None)
    return beta_hat

def compute_prediction(beta,Xi):
    prediction = beta[0] + beta[1]*Xi + beta[2]*Xi**2 + beta[3]*Xi**3 + beta[4]*Xi**4
    return prediction

def support_len(X):
    return np.max(X)-np.min(X)

def optimal_bandwidth(n,alpha,beta,sigma_2,N):

    X,Y=data_generation(n,alpha,beta,sigma_2)
    blocks_X, blocks_Y = blocks_creator(X,Y,N)

    theta22_hat = 0.0
    RSS = 0.0

    for i in range(N):
        Xi=blocks_X[i]
        Yi=blocks_Y[i]
        beta_hat=ols_etimate(Xi,Yi)
        diff_2_m=compute_diff_2_m(beta_hat,Xi)
        theta22_hat+=np.sum(diff_2_m**2)
        prediction=compute_prediction(beta_hat,Xi)
        RSS += np.sum((Yi - prediction)**2)

    theta22_hat = theta22_hat / n
    sigma2_hat = RSS / (n - 5*N)
    h_AMISE = n**(-1/5) * (35 * sigma2_hat * support_len(X) / theta22_hat)**(1/5)

    return h_AMISE, X, Y, theta22_hat, sigma2_hat, RSS, support_len(X)

def compute_N_optimal(n,alpha,beta,sigma_2,N_list):

    X,Y=data_generation(n,alpha,beta,sigma_2)
    RSS=[]
    for j,N in enumerate(N_list):
        blocks_X, blocks_Y = blocks_creator(X,Y,N)
        RSS.append(0.0)
        for i in range(N):
            Xi=blocks_X[i]
            Yi=blocks_Y[i]
            beta_hat=ols_etimate(Xi,Yi)
            prediction=compute_prediction(beta_hat,Xi)
            RSS[j] += np.sum((Yi - prediction)**2)

    N_max = int(np.max([np.min([np.floor(n/20), 5]), 1]))
    blocks_X, blocks_Y = blocks_creator(X,Y,N_max)
    RSS_Nmax=0.0
    for i in range(N_max):
        Xi=blocks_X[i]
        Yi=blocks_Y[i]
        beta_hat=ols_etimate(Xi,Yi)
        prediction=compute_prediction(beta_hat,Xi)
        RSS_Nmax += np.sum((Yi - prediction)**2)

    C_p=RSS/(RSS_Nmax/(n-5*N_max))-(n-10*np.array(N_list))
    idx_opt=np.argmin(C_p)
    N_opt=N_list[idx_opt]

    return N_opt
