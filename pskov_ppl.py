
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd

# data to generate main sample
N=50000
male_percent = 0.45
female_percent  = 1 - male_percent
gender_ratio = [male_percent, female_percent]

m_male = 178
m_female = 174
std_male = 5.2
std_female = 5.4

# data generator function
def Generate(N, gender_ratio, m_male, m_female, std_male, std_female)->pd.DataFrame:
    gender = np.random.choice(['male', 'female'], N, p=gender_ratio)
    h = np.zeros(N)
    for i in range(N):
        if gender[i] == 'male':
           h[i] = np.random.normal(m_male, std_male)
        else:
            h[i] = np.random.normal(m_female, std_female)
    df = pd.DataFrame(columns=['Gender','Height'])
    df.Gender = gender 
    df.Gender = df.Gender.astype('category')
    df.Height = h
    return df

# generate data
df = Generate(N, gender_ratio, m_male, m_female, std_male, std_female)

# print some information
print(df.shape)
print(df.head(10))

males = df[df.Gender == 'male']
females = df[df.Gender == 'female']

print(males.shape)
print(females.shape)
print(males.head(10))
print(females.head(10))

# descriptive statistics
def get_stats(data):
    count  = len(data)
    min = np.min(data)
    max = np.max(data)
    range = max - min
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75) 
    iqr = q3 - q1   
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    se = np.sqrt(var) / np.sqrt(count)   # standard error
    z95 = 1.645                          # 95% confidence level
    ci = z95 * se                         # 95% CI
    mean_from = mean - ci                 # mean from CI lower limit
    mean_to = mean + ci                   # mean to CI upper limit
    chi95 = stats.chi2.ppf(0.95, count)  # 95% confidence level for chi-square test of independence
    chi5 = stats.chi2.ppf(0.05, count)
    # confidence limits for variance
    var_from = var / chi95 * count               # Variance from 95% CL (lower limit)
    var_to = var / chi5 * count                   # Variance to 95% CL (upper limit)
    # for std dev
    std_from = np.sqrt(var_from)               # Std dev from 95% CL (lower limit)
    std_to = np.sqrt(var_to)                  # Std dev to 95
    # skewness
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    res = {'count':count,
            'min':min,'max':max,'range':range,'median':median,'q1':q1,'q3':q3,'iqr':iqr,
            'mean':mean,'variance':var,'std':std, 
            'mean_fom': mean_from, 'mean_to': mean_to,'se':se,
            'var_from':var_from,'var_to':var_to,'std_from':std_from,'std_to':std_to,
            'skewness':skewness,'kurtosis':kurtosis}
    return res

def print_stats(stats,title='Descriptive Stats'):
    """Print out the statistical measures in a nice format"""
    print('\n'+'='*len(title)*2)
    print(title.center(len(title)*2))
    print('='*len(title)*2)
    fmt = "{:>12} => {:<28}"
    print(fmt.format("Measure","Value"))
    print('-'*50)
    for key,value in (stats.items()):
        print(fmt.format(key,value))
    print('-'*50)
    print('\n')

def plot_hist_boxplot(data,title=''):
    figure,axes=plt.subplots(2,1,figsize=(8,4),                    
                sharex=True,
                gridspec_kw=dict(height_ratios=[3,0.5]))
    sns.histplot(x=data, kde=True,ax=axes[0], stat='density')
    # plot normal distribution
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(np.mean(data)-3.5*sigma, np.mean(data)+3.5*sigma, 100)
    axes[0].axvline(np.min(data),linestyle='dotted')
    axes[0].axvline(np.max(data),linestyle='dotted')
    axes[0].axvline(np.mean(data),linestyle='dotted', color = 'green')
    axes[0].plot(x, stats.norm.pdf(x,mu,sigma), linewidth=2, color='orange')
    axes[0].set_title(title)
    label=f'mean={np.mean(data):.3f},\nsd={np.std(data):.3f},\nmedian={np.median(data):.3f}\nN={len(data)}'
    axes[0].text(0.01, 0.75, label, horizontalalignment='left',transform=axes[0].transAxes,fontsize=10)
    sns.boxplot(x=data, showmeans=True, ax=axes[1])
    # axes[1].set_title('Boxplot')
    plt.tight_layout()
    plt.show()

def Report(data,var_name=''):
    if var_name == '' or var_name is None:
        var_name = 'Data'
    stat = get_stats(data)
    print_stats(stat,"Data Description of "+var_name)
    plot_hist_boxplot(data,"Histogram & Box Plot of "+var_name)
    return stat

Report(df.Height,'height overall')
Report(males.Height,'height of M')
Report(females.Height,'height of F')

sns.histplot(df.Height,fill=False, common_bins=False,common_norm=False,stat='density',kde=True)
sns.histplot(data=df, x='Height',hue='Gender', fill=False, common_bins=False,common_norm=False,stat='density')
plt.show()

sns.boxplot(data=df, x='Height',y='Gender',hue='Gender')
plt.show()

sns.violinplot(data=df, x='Height',y='Gender',hue='Gender')
plt.show()

mu = 175.8
result = stats.ttest_1samp(df.Height,mu)
print(100*'=')
print(result)
t,p = result[0],result[1]
print(f'Test result is: {result[0]:.4f} with p-value {result[1]:.4f}')
print(50*'-')
# If the p value is less than 0.05 then reject null hypothesis and conclude that there is a significant difference
# between the population mean and the sample mean.
# If the p value is greater than 0.05 then accept null hypothesis and conclude that there is no significant difference
# between the population mean and the sample mean.
if p < 0.05:
    print("Reject null hypothesis: There is a significant difference between the population mean and the sample mean")
else:
    print("Accept null hypothesis: There is no significant difference between the population mean and the sample mean")
print(100*'=')

result = stats.ttest_ind(males.Height, females.Height)  
print(100*'=')
print(result)
t,p = result[0],result[1]
print(f'Test result is: {result[0]:.4f} with p-value {result[1]:.4f}')
print(50*'-')
# If the p value is less than 0.05 then reject null hypothesis and conclude that there is a significant difference
# between the population mean and the sample mean.
# If the p value is greater than 0.05 then accept null hypothesis and conclude that there is no significant difference
# between the population mean and the sample mean.
if p < 0.05:
    print("Reject null hypothesis: There is a significant difference between the means in samples")
else:
    print("Accept null hypothesis: There is no significant difference between the  means in samples")
print(100*'=')

result = stats.f_oneway(males.Height, females.Height)  
print(100*'=')
print(result)
F,p = result[0],result[1]
print(f'Test result is: {result[0]:.4f} with p-value {result[1]:.4f}')
print(50*'-')
# If the p value is less than 0.05 then reject null hypothesis and conclude that there is a significant difference
# between the population mean and the sample mean.
# If the p value is greater than 0.05 then accept null hypothesis and conclude that there is no significant difference
# between the population mean and the sample mean.
if p < 0.05:
    print("Reject null hypothesis: There is a significant difference between the standard deviations in samples")
else:
    print("Accept null hypothesis: There is no significant difference between the standard deviations in samples")
print(100*'=')