
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# define gamma distibution function using scipy as sp
import scipy as sp
from scipy.stats import gamma, beta

# calculate theoretical gamma density function
def gamma_density(x, alpha, beta):
    return gamma.pdf(x, alpha, scale=1.0/beta)

#generate random values 
def generate_gamma(N, alpha, beta):
    return np.random.gamma(alpha, 1.0/beta, N)

# example usage:
N = 10000
alpha = 10
beta = 1
values = generate_gamma(N, alpha, beta)

# plot histogram of and theoretical curve of gamma distribution using seaborn
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.histplot(values, kde=False, stat='probability')

x = np.linspace(0, max(values), N)
y =gamma_density(x,alpha,beta)
sns.lineplot(x=x, y=y,c='orange',ax=ax1)
plt.show()

# generate N Beta distributed random value with alpha and beta parameters
def generate_beta(N, alpha, beta):
    return np.random.beta(alpha, beta, N)

# calculate theoretical beta density function
def beta_density(x, alpha, beta):
    return sp.stats.beta.pdf(x, alpha, beta)

# example usage:
N = 10000
alpha1 = 6
beta1 = 4

values = generate_beta(N, alpha1, beta1)

# plot histogram of and theoretical curve of beta distribution using seaborn
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.histplot(values, kde=False, stat='probability')
x = np.linspace(0, 1, N)

y = beta_density(x, alpha1, beta1)

sns.lineplot(x=x, y=y,c='orange',ax=ax1)

plt.show()


