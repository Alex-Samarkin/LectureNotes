import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# generate N uniform distributed random value between A and B
# plot scatterplot on one half plot and horisontally oriented histogramm on second
# define function and use seaborn as sns for plot
def scatter_hist(N,A,B):

    # generate N uniform distributed random value between A and B
    data = np.random.uniform(A, B, N)

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle(f"Scatterplot and historamm of {N} data [{A};{B}]")

    # plot scatterplot on one half plot
    sns.scatterplot(x=np.arange(N),y=data, ax=axes[0])

    # plot histogramm on second
    sns.histplot(y=data, ax=axes[1],stat='percent')
    plt.show()
    return data


# example usage
N = 1000

A = 0
B = 1

scatter_hist(N,A,B)
