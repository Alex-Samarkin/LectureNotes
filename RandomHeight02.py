import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# generate N normally distributed value with m as average and sd as standard deviation as column of pandas dataframe
def random_height(N,m,sd):
    data = pd.DataFrame({'Height': np.random.normal(m, sd, N)})
    return data

# generate 1000 simulated data
N = 1000
m = 174.0
sd = 8.0

data = random_height(N,m,sd)
data['X'] = np.arange(N)

sns.scatterplot(data=data,x='X',y='Height')
plt.show()

y = data['Height'].values
data['Sorted'] = np.sort(y)
sns.scatterplot(data=data,y='X',x='Sorted')
plt.show()

print(data.iloc[200,:])

sns.histplot(data=data,x='Sorted',cumulative=True)
plt.show()
sns.kdeplot(data=data,x='Sorted',fill=True,cumulative=True)
plt.show()