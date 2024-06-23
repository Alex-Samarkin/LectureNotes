import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

N=1500
unsorted = np.random.randn(N)
sorted = np.sort(unsorted)

fig,axes = plt.subplots(1,3,figsize=(12,6))
sns.scatterplot(y=np.arange(N),x=unsorted,ax=axes[0])
axes[0].set_title('Unordered')
sns.scatterplot(y=np.arange(N),x=sorted,ax=axes[1])
axes[1].set_title('Ordered')
sns.histplot(unsorted,ax=axes[2],stat='density')
axes[2].set_title('Histogram')
# plot normal curve use scipy 

x = np.linspace(-3,3,100)
y= sp.stats.norm.pdf(x)
sns.lineplot(x=x,y=y,color='orange')

plt.show()