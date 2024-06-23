import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# generate N uniform distributed random value between A and B
# plot scatterplot on one half plot and horisontally oriented histogramm on second
# define function and use seaborn as sns for plot
def scatter_hist(N,Lambda,k=1):

    # generate N uniform distributed random value between A and B
    data=np.zeros(N)
    for i in range(k):
        data = data+np.random.exponential(Lambda, N)

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle(f"Scatterplot and historamm of {N} data with lambda {Lambda} (sum of {k})")

    # plot scatterplot on one half plot
    sns.scatterplot(x=np.arange(N),y=data, ax=axes[0])

    # plot histogramm on second
    sns.histplot(y=data, ax=axes[1],stat='percent')
    plt.show()
    return data

for i in [1,2,3,5,12,35]:
    scatter_hist(5000,1,i)
