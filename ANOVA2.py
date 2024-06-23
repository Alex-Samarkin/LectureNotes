import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd

rng = np.random.default_rng()
# generate one of the symbol from string using fixed probability lidst
def generate_string(lidst,p=[0.5,0.5])->str:
    return rng.choice(lidst, 1, p)

# generate male or female according current state inside populatioin
def MaleFemale()->str:
    return generate_string(['Male','Female'],[0.45,0.55])[0]

# generate height according current state inside populatioin based on gender and normal law
def Height(gender)->float:
    if gender == 'Male':
        return rng.normal(178, 8)
    else:
        return rng.normal(169, 8)

# generate age from 20 to 80 
def Age()->int:
    return rng.integers(20, 80)

# find age group (lesser than 40 - 1, 40-60 - 2, 60-80 - 3, bigger - 4)
def AgeGroup(age)->int:
    if age < 40:
        return 1
    elif age < 60:
        return 2
    elif age < 80:
        return 3
    else:
        return 4

# generate diabetes at 30% cases
def Diabetes()->str:
    return generate_string(['Yes','No'],[0.4,0.6])[0]    

# generate weight, based on height, gender and diabet presence
def Weight(gender, height, diabetes)->float:
    if diabetes == 'Yes':
        if gender == 'Male':
            return height * 0.9
        else:
            return height * 0.8
    else:
        if gender == 'Male':
            return height * 0.8
        else:
            return height * 0.7
        
#generate glucose level, based on diabete presence
def Glucose(diabetes)->float:
    if diabetes == 'Yes':
        return rng.normal(15.0, 2.0)
    else:
        return rng.normal(7.5, 1.0)

# generate pandas dataframe with columns Id, Gender,Age,AgeGroup, Height, Weight, Glucose, Diabet
def generate_dataframe(N):
    df = pd.DataFrame(columns=['Id','Gender','Age','AgeGroup','Height','Weight','Glucose','Diabet'])
    for i in range(N):
        df.loc[i,'Id'] = int(i)
        df.loc[i,'Diabet'] = Diabetes()
        df.loc[i,'Gender'] = MaleFemale()
        df.loc[i,'Age'] = Age()
        df.loc[i,'AgeGroup'] = AgeGroup(df.loc[i,'Age'])
        df.loc[i,'Height'] = Height(df.loc[i,'Gender'])
        df.loc[i,'Weight'] = Weight(df.loc[i,'Gender'],df.loc[i,'Height'],df.loc[i,'Diabet'])
        df.loc[i,'Glucose'] = Glucose(df.loc[i,'Diabet'])
    return df

# generate random N from 120 to 500
N = rng.integers(120,500)
df = generate_dataframe(N)
print(df.head(10))

# plott dtypes of all columns
print(df.dtypes)
# convert all types to numeric
df['Id'] = pd.to_numeric(df['Id'])
df['Height'] = pd.to_numeric(df['Height'])
df['Weight'] = pd.to_numeric(df['Weight'])
df['Glucose'] = pd.to_numeric(df['Glucose'])
df['Age'] = pd.to_numeric(df['Age'])
df['AgeGroup'] = pd.to_numeric(df['AgeGroup'])
print(df.dtypes)


import matplotlib.pyplot as plt
import seaborn as sns

# plot histogram using seaborn
sns.histplot(df, x='Glucose', kde=True, stat='probability', palette='Set1')
plt.show()
sns.histplot(df, x='Glucose', hue='Gender', kde=True, stat='probability', palette='Set1')
plt.show()
sns.histplot(df, x='Glucose', hue='Diabet', kde=True, stat='probability', palette='Set1')
plt.show()

# plot boxplot of data using seaborn and categorial columns as group variable
sns.boxplot(x='Gender', y='Glucose', data=df)
plt.show()
sns.boxplot(x='Diabet', y='Glucose', data=df)
plt.show()
sns.boxplot(x='Diabet', y='Glucose', hue='AgeGroup', data=df)
plt.show()

# plot interaction plot from sttatsmodels library
from statsmodels.graphics.factorplots import interaction_plot
fig = interaction_plot(x=df['Diabet'].astype('str'), trace=df['Gender'].astype('str'),response=df['Glucose'],func=np.mean, colors=['red','blue'], markers=['D','^'], ms=10) 

plt.show()
fig = interaction_plot(x=df['AgeGroup'].astype('str'), trace=df['Gender'].astype('str'),response=df['Glucose'],func=np.mean, colors=['red','blue'], markers=['D','^'], ms=10) 

plt.show()

# perform ANOVA test
import statsmodels.api as sm
from statsmodels.formula.api import ols

df.loc[:,'Gender'] = df['Gender'].astype("category")
df['Diabet'] = df['Diabet'].astype("category")    
df['AgeGroup'] = df['AgeGroup'].astype("category")

print(df.dtypes)

model = ols('Glucose ~ Diabet + Gender + AgeGroup + Diabet : Gender + Diabet : AgeGroup + Gender : AgeGroup + Diabet : Gender : AgeGroup - 1', data=df).fit()
print(model.summary())

aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

# perform ANOVA post-hoc test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

mc = MultiComparison(df['Glucose'], df['Diabet'])
print(mc.groupsunique)
result = mc.tukeyhsd()
print(result.summary())

#plot post-hoc test results
result.plot_simultaneous()
plt.show()
result.plot_simultaneous(comparison_name='Yes')
plt.show()

df['combined_group'] = df['Diabet'].astype(str) + "_" + df['Gender'].astype(str) + "_" + df['AgeGroup'].astype(str)
mc = MultiComparison(df['Glucose'], df['combined_group'])
print(mc.groupsunique)
result = mc.tukeyhsd()
print(result.summary())

#plot post-hoc test results
result.plot_simultaneous()
plt.show()
result.plot_simultaneous(comparison_name='Yes_Male_1')
plt.show()


