# generate two normally distributed samples wit same data inside    

import numpy as np
import scipy as sp
from scipy import stats
# generate two normally distributed samples wit same data inside

mu1 = -1
mu2 = 1
sigma = 1

N = 52

x = np.random.normal(mu1, sigma, N)
y = np.random.normal(mu2, sigma, N)
z = np.random.normal(mu1*1.05, sigma, N)

# add x  and y as column to pandas dataframe and add second column with x and y  
import pandas as pd
df = pd.DataFrame({'x':x, 'y':y, 'z':z})
print(df.head())
df = df.stack()
df = df.reset_index().rename(columns={'level_1': 'xyz',0:'Data'}).drop(columns={'level_0'})

print(df.head(10))

# plot histogram using seaborn
import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(df, x= range(0,3*N),y='Data', hue='xyz', palette='Set1')
plt.show()
sns.histplot(df, x='Data', kde=True, stat='count', palette='Set1')
plt.show()
sns.histplot(df, x='Data', hue='xyz', kde=True, stat='count', palette='Set1')
plt.show()
sns.boxenplot(df, x='Data', hue='xyz', palette='Set2')
plt.show()
sns.violinplot(df, x='Data', hue='xyz', palette='Set3')
plt.show()

# calculate statistics
print("Means by group")
print(df.groupby(['xyz']).mean().unstack())
print("Variances by group")
print(df.groupby(['xyz']).var().unstack())
print("Descriptive stats")
print(df.groupby(['xyz']).describe().unstack().unstack())

# perform AD test for normality
print("Ad test for normality")
print(sp.stats.anderson(df['Data'], dist='norm'))

# perform t-test
print("T-test for equal means xy")
print( sp.stats.ttest_ind( df['Data'][df['xyz']=='x'],df['Data'][df['xyz']=='y']) )
print( "T-test for equal means xz")
print( sp.stats.ttest_ind(df['Data'][df['xyz']=='x'],df['Data'][df['xyz']=='z']) )
print( "T-test for equal means yz")
print( sp.stats.ttest_ind(df['Data'][df['xyz']=='y'],df['Data'][df['xyz']=='z']) )

# perform ANOVA test
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Data ~ C(xyz)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)











