import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as sts
import pandas as pd

# Generate random data according to an normal distribution and amount of random N
N=np.random.randint(25,75) # Number of observations (samples) to generate
data = sts.norm.rvs(size=N)
# Calculate the mean and standard deviation
mean = np.mean(data)
std = np.std(data)

# Plot the data aas relative freq histogram 
plt.hist(data, density=True)
plt.title( f"Mean is {mean:.3f} Std is {std:.3f} N={N}")

# Plot normal distribution
x = np.linspace(min(data), max(data), N)
plt.plot(x, sts.norm.pdf(x, mean, std))

plt.show()

# 2. Q-Q plot
plt.figure()
plt.title( f"Mean is {mean:.3f} Std is {std:.3f} N={N}")
sts.probplot(data, dist="norm", plot=plt)
plt.show()

# 3. Shapiro-Wilk test
print(sts.shapiro(data))
# The test statistic is a two-sided p-value.
# The p-value is the probability of obtaining a test statistic at least as extreme as the one that was actually observed, given the null hypothesis.
# The p-value is a probability that the data is drawn from the distribution that the null hypothesis claims.
# The p-value is less than 0.05, so the data is unlikely to come from a normal distribution.

# 4. Anderson-Darling test
print(sts.anderson(data))

# 5. Kolmogorov-Smirnov test
print(sts.kstest(data, 'norm'))

# 6. KS-Lilleforce test
print(sts.kstest(data, 'norm', alternative='less'))

# 7. Chi-square test
print(sts.chisquare(data))
