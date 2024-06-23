import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

p=0.3
N=1000
r = sp.stats.bernoulli.rvs(p, size=N)
sns.histplot(r,discrete=True)
plt.show()

n=12
p=0.5
r1 = sp.stats.binom.rvs(n,p,size=N)
sns.histplot(r1,discrete=True)
plt.show()

mu1=10.0
mu2=6
mu3=2
r21 = sp.stats.poisson.rvs(mu1,size=N)
r22= sp.stats.poisson.rvs(mu2,size=N)
r23 = sp.stats.poisson.rvs(mu3,size=N)
sns.histplot(r21,discrete=True,element="step", fill=False,label="mu=10")
sns.histplot(r22,discrete=True,element="step", fill=False,label="mu=6")
sns.histplot(r23,discrete=True,element="step", fill=False,label="mu=2")
plt.legend()
plt.show()

p=0.1
N=1000
r3=sp.stats.geom.rvs(p,size=N)
sns.histplot(r3,discrete=True)
plt.show()

N=1000
M=1000
n=200
K=25
r4 = sp.stats.hypergeom.rvs(M, n, K, size=N)
sns.histplot(r4,discrete=True)
plt.show()