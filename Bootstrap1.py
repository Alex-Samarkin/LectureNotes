import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_stats(data):
    median = np.median(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    skewness =sp.stats.skew(data)
    kurtosis = sp.stats.kurtosis(data)
    stats = {'median':median,'mean':mean,'variance':var,'std':std,'skewness':skewness,'kurtosis':kurtosis}
    return stats
# function to generate N randomly distributed data with specific distribution and return numpy array
# distr_law must be from uniform,normal,exponential or normal if other
def random_data(N=10000,distr_law="normal",mean=0,sd=1,a=-3,b=3,scale=1):
    if not(distr_law in ['uniform','normal','exponential']):
        distr_law = 'normal'
        pass
    res=np.zeros(N)
    if distr_law=='uniform':
        res = np.random.uniform(a,b,N)
        pass
    if distr_law=='normal':
        res = np.random.normal(mean,sd,N)
        pass
    if distr_law=='exponential':
        res = np.random.exponential(scale,N)
        pass

    stats = get_stats(res)
    return res,stats

def plot_hist_boxplot(data,title=''):
    figure,axes=plt.subplots(2,1,figsize=(8,4),                    
                sharex=True,
                gridspec_kw=dict(height_ratios=[3,0.5]))
    sns.histplot(x=data, kde=True,ax=axes[0])
    axes[0].set_title(title)
    label=f'mean={np.mean(data)},\nsd={np.std(data)},\nmedian={np.median(data)}'
    axes[0].text(0.01, 0.75, label, horizontalalignment='left',transform=axes[0].transAxes,fontsize=10)
    sns.boxplot(x=data, showmeans=True, ax=axes[1])
    # axes[1].set_title('Boxplot')
    plt.tight_layout()
    plt.show()

data,stats = random_data(distr_law='exponential',scale=2)
print(stats)
plot_hist_boxplot(data,'General population')

def random_choice_plot(data,N1=300):
    res = np.random.choice(data,N1,replace=True)
    plot_hist_boxplot(res,f'Random sample')
    pass

for k in range(4):
    random_choice_plot(data)

df = pd.DataFrame()

mx=[0.0]
mx.append(np.abs(np.mean(data)-2))
for i in range(2,17):
    n1 = (int)(len(data) / (2**(i-1)))
    mx.append(np.abs(np.mean(np.random.choice(data,n1,replace=True))-2))
    pass
sns.barplot(x=np.arange(len(mx)),y=mx,)
plt.show()

def bootstrap(data,df,N1=300,k=500):
    res_s = []
    for i in range(k):
        res = np.random.choice(data,N1,replace=True)
        stats = get_stats(res)
        res_s.append(stats)
        pass
    df = pd.DataFrame(res_s)
    return df

df = bootstrap(data,df,N1=1000,k=1000)
print(df.head())

def plot_bootstrap(df):
    columns = df.columns
    for c in columns:
        plot_hist_boxplot(df[c],f'{c} distribution')

plot_bootstrap(df)