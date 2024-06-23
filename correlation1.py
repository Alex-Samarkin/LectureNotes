import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# generate  normally disributed numpy array x
N = 250
x = np.random.normal(0, 1, N)
# generate normally distributed numpy array y with mean 0 and variance 1
y = np.random.normal(0, 1, N)
# calculate the sum of x and y
s = x + y

# plot scatterplot of x, y and s against index using seaborn
sns.scatterplot(x=range(N), y=x)
sns.scatterplot(x=range(N), y=y)
sns.scatterplot(x=range(N), y=s)
plt.show()

# calc correlation between x and y and s and print and plot matrix of it
print(np.corrcoef([x, y, s]))
sns.heatmap(np.corrcoef([x, y, s]), annot=True)
plt.show()

sns.pairplot(pd.DataFrame({'x': x, 'y': y, 's': s}), diag_kind='kde')
plt.show()

sns.pairplot(pd.DataFrame({'x': x, 'y': y, 's': s}), diag_kind='hist')
plt.show()

# produce normally random rho from  0.9 to 1.1  and uniform distributed phi from 0 to 2pi
rho = np.random.uniform(0.9, 1.1, N)
phi = np.random.uniform(0, 2 * np.pi, N)
# calculate x and y from rho and phi
x = rho * np.cos(phi)
y = rho * np.sin(phi)

# plot scatterplot of x and y
sns.scatterplot(x=x, y=y)
plt.show()

# calculate correlation between x and y and print and plot matrix of it
print(np.corrcoef([x, y]))  # correlation between x and y
sns.heatmap(np.corrcoef([x, y]), annot=True)
plt.show()
sns.pairplot(pd.DataFrame({'x': x, 'y': y}), diag_kind='kde')
plt.show()

