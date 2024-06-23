import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

N=1500
rand_n = np.random.randn(N)
rand_uniform = np.random.uniform(-3,3,N)
rand_exp = np.random.standard_exponential(N)

def k_hist(data,titles):
    k = len(data)
    fig,axes = plt.subplots(1,k,figsize=(12,6))
    for i in range(k):
        sns.histplot(data[i],kde=True,stat='density', ax=axes[i])
        axes[i].set_title(titles[i])
        pass
    plt.show()
    pass

k_hist([rand_n,rand_uniform,rand_exp],['Normal','Uniform','Exponential'])

# descriptive stats for np array
def print_stat(rand_n):
    print(80*'=')
    print("Descriptive Stats:")
    print(80*'=')

    print('Parametric stats:')
    print(80*'-')
    print("Mean: ",np.mean(rand_n))
    print("Median: ",np.median(rand_n))
    print("Variance: ",np.var(rand_n))
    print("Standard Deviation: ",np.std(rand_n))
    print("Skewness: ",sp.stats.skew(rand_n))
    print("Kurtosis: ",sp.stats.kurtosis(rand_n))

    print(80*'-')
    print('NonParametric stats:')
    print(80*'-')
    # min max range quartiles, deciles and interquartile range
    print("Min: ",np.min(rand_n))
    print("Max: ",np.max(rand_n))
    print("Range: ",np.max(rand_n)-np.min(rand_n))
    print("Quartiles: ",np.quantile(rand_n,[0.25,0.5,0.75]))
    print("Deciles: ",np.quantile(rand_n,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
    print("Interquartile range: ",np.quantile(rand_n,[0.75])-np.quantile(rand_n,[0.25]))  
    print(80*'=')

print_stat(rand_n=rand_n)
print()
print_stat(rand_n=rand_uniform)
print()
print_stat(rand_n=rand_exp)
print()

# define a function to plot q-q graph, based on np array x and normal distribution using seaborn
def qq_plot(x):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Q-Q Plot ')
    ax = sp.stats.probplot(x, dist=sp.stats.norm, plot=ax)
    plt.show()
    return None
# compare the distributions of uniform random numbers and exponential random numbers with standard Normal Distribution
qq_plot(rand_n)
qq_plot(rand_uniform)
qq_plot(rand_exp)

def hist_plot(x):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Histogram ')
    sns.histplot(x, kde=False, stat='density', ax=ax)
    mean = np.mean(x)
    std = np.std(x)
    x_min = min(x)
    x_max = max(x)
    x1 = np.linspace(x_min,x_max, int(np.log2(len(x))+1)*8)
    sns.scatterplot(x=x1, y=sp.stats.norm.pdf(x1),label='norm pdf')
    plt.show()
    return None

hist_plot(rand_n)
hist_plot(rand_uniform)
hist_plot(rand_exp)

sns.boxplot(x=rand_exp, showmeans=True)
plt.show()
sns.boxplot(x=rand_n, showmeans=True)
plt.show()

def hist_box(x):
    # specify plot layouts with different width using subplots()
    fig, axes = plt.subplots(1,2,
                      figsize=(12,6),
                      sharey=True,
                      gridspec_kw=dict(width_ratios=[3,0.5]))
    sns.histplot(y=x, kde=True,ax=axes[0])
    axes[0].set_title('Histogram')
    sns.boxplot(y=x, showmeans=True, ax=axes[1])
    axes[1].set_title('Boxplot')
    plt.show()

hist_box(rand_n)
hist_box(rand_exp)