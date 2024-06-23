import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as sts
import pandas as pd

# Generate random data according to an exponential distribution
N=np.random.randint(100,10000) # Number of observations (samples)
scale=np.random.uniform(1,10) # Scale parameter of exponential distribution
data = sts.expon.rvs(scale=scale, size=N)
# real scale
scale1=np.mean(data)
print(N,scale,scale1)

#Sturges rule
K=int(np.log2(N)+1)
bins=K # number of bins is twice the number of Sturges

# Plot histogram using seaborn
sns.histplot(data, kde=False, color='skyblue', stat='density', bins=bins, label='Observed')

# Overlay theoretical exponential distribution
x = np.linspace(0, max(data), N)  # Generate x values for the theoretical distribution plot
plt.plot(x, sts.expon.pdf(x, scale=scale1), 'r-', label='Theoretical')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram of Exponential Distribution')
plt.legend()

# Show plot
plt.show()

# Q-Q plot of exponential distribution against observed data
fig, ax = plt.subplots()
sts.probplot(data, dist=sts.expon(scale=scale1),fit=True,plot=ax)

# Add labels and title
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.title('Q-Q Plot of Exponential Distribution')
plt.show()

pd.set_option("mode.copy_on_write", False)
pd.option_context('mode.use_inf_as_na', True)

df = pd.DataFrame()
df1 = pd.DataFrame()
df1['data']=data
#expected
bin_edges = np.histogram_bin_edges(data,bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
df['Left Edge'] = bin_edges[:-1]
df['Right Edge'] = bin_edges[1:]
df['Mid of bin'] = bin_centers

# observed data in every bin
df['Observed'] = np.histogram(data, bins)[0]
df['Observed rel'] = df['Observed']/df['Observed'].sum()
df['Observed cumulative'] = df['Observed'].cumsum()
df['Observed cumulative rel'] = df['Observed rel'].cumsum()
# theoretical data in every bin
df['Theoretical cumulative rel'] = sts.expon.cdf(bin_edges[1:], scale=scale1)
t=df.loc[:,'Theoretical cumulative rel']
t[len(t)-1]=1
df.loc[:,'Theoretical cumulative rel'] = t
df['Theoretical cumulative'] = df['Theoretical cumulative rel']*df['Observed'].sum()
t = df.loc[:,'Theoretical cumulative rel']
t1 = t.diff()
t1[0] = t[0]
df['Teoretical rel'] =t1
s = df.loc[:,'Observed'].sum()
t = df.loc[:,'Teoretical rel']
t1 = t*s
df['Teoretical'] = t1

t=df.loc[:,'Teoretical']
t1=df.loc[:,'Observed']
df['Difference'] = t-t1
df['Difference^2'] = df['Difference']**2

sns.histplot(df['Difference'])
plt.show()
print(df)

#chi-square test of goodness of fit
hi2_stat, p_val = sts.chisquare(df['Observed'], f_exp=df['Teoretical'],)
print(hi2_stat, p_val)
if p_val < 0.05:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")

