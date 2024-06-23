
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# x value from -5 to 5
x = np.linspace(-5, 5, 1000)

# Generate data for the normal distribution
mu = 0
sigma = 1

# PDF and CDF for normal distribution
pdf_norm = stats.norm.pdf(x, mu, sigma)
cdf_norm = stats.norm.cdf(x, mu, sigma)

# two subplots sharing the x-axis for PDF and CDF
figure,axes=plt.subplots(2,1,figsize=(12,6), sharex=True)

# Generate data for the Student's t-distribution
for df in range(1,9):
    pdf_t = stats.t.pdf(x, df)
    cdf_t = stats.t.cdf(x, df)
    # Plot both PDFs on the same graph
    sns.lineplot(x=x, y=pdf_t, ax=axes[0],label=f'Student\'s t-distribution ({df})')
    sns.lineplot(x=x, y=cdf_t, ax=axes[1],label=f'Student\'s t-distribution ({df})',)


# plot the normal distribution on the same graph as the Student's t-distribution
sns.lineplot(x=x, y=pdf_norm, label='Normal distribution',ax=axes[0])
axes[0].grid()
axes[0].set_title('PDF')
sns.lineplot(x=x, y=cdf_norm, label='Normal distribution',ax=axes[1])
axes[1].grid()
axes[1].set_title('CDF')
plt.legend()

plt.show()

# two subplots sharing the x-axis for PDF and CDF
figure,axes=plt.subplots(2,1,figsize=(12,6), sharex=True)

# Generate difference data for the Student's t-distribution
for df in range(1,9):
    pdf_t1 = stats.t.pdf(x, df) - pdf_norm
    cdf_t1 = stats.t.cdf(x, df) - cdf_norm
    # Plot both PDFs on the same graph
    sns.lineplot(x=x, y=pdf_t1, ax=axes[0],label=f't-distribution ({df}) - norm(0,1)')
    sns.lineplot(x=x, y=cdf_t1, ax=axes[1],label=f't-distribution ({df}) - norm(0,1)')


axes[0].grid()
axes[0].set_title('Difference between PDF')
axes[1].grid()
axes[1].set_title('Difference between CDF')
plt.legend()

plt.show()

# Generate a random NumPy array
data = np.random.normal(loc=0, scale=1, size=100)

# Calculate the confidence interval for the mean
def calculate_confidence_interval(data, confidence_level=0.95):
    confidence_level = 0.95
    degrees_of_freedom = len(data) - 1
    mean = np.mean(data)
    standard_error = stats.sem(data)
    confidence_interval = stats.t.interval(confidence_level, degrees_of_freedom, mean, standard_error)
    return confidence_interval

# Print the confidence interval
conf_level = 0.95
ci = calculate_confidence_interval(data,confidence_level=conf_level)
print(f"The {conf_level*100}% confidence interval for the mean is [{ci[0]:.2f}, {ci[1]:.2f}].")

sns.boxplot(x=data,showmeans=True)
plt.hlines(0,ci[0],ci[1],color='red')
plt.axvline(x=ci[0], color='r', linestyle='--')
plt.axvline(x=ci[1], color='r', linestyle='--')
plt.title(ci)
plt.suptitle(f"Mean is {np.mean(data)}")
plt.show()

# Calculate the confidence interval for the variance
def calculate_confidence_interval_2(data, confidence_level=0.95):
    degrees_of_freedom = len(data) - 1
    mean = np.mean(data)
    standard_error = stats.sem(data)
    sample_variance = np.var(data)
    # Calculate the confidence interval for the sample_variance using chi square distribution
    chi2 = stats.chi2.ppf((1 + confidence_level)/2, degrees_of_freedom)
    confidence_interval = [sample_variance - chi2/degrees_of_freedom * standard_error,
                           sample_variance + chi2/degrees_of_freedom * standard_error]
    return confidence_interval

ci2 = calculate_confidence_interval_2(data,confidence_level=conf_level)
def calculate_confidence_interval_1(data, confidence_level=0.95):
    ci = np.sqrt(calculate_confidence_interval_2(data,conf_level))
    return ci
ci1 = calculate_confidence_interval_1(data,confidence_level=conf_level)
print(f"The {conf_level*100}% confidence interval for the varaiance {np.var(data):.2f} is [{ci2[0]:.2f}, {ci2[1]:.2f}].")
print(f"The {conf_level*100}% confidence interval for the std dev {np.var(data):.2f} is [{ci1[0]:.2f}, {ci1[1]:.2f}].")

sns.boxplot(x=data,showmeans=True)
plt.hlines(0,ci[0],ci[1],color='red')
plt.axvline(x=ci[0], color='r', linestyle='--')
plt.axvline(x=ci[1], color='r', linestyle='--')
plt.suptitle(f"Mean [{ci[0]:.2f} <-- {np.mean(data):.2f} -->{ci[1]:.2f}]") 

# Add a lines to the plot
plt.hlines(0,ci1[0],ci1[1],color='blue')
plt.axvline(x=ci1[0], color='blue', linestyle='--')
plt.axvline(x=ci1[1], color='blue', linestyle='--')
plt.title(f"Std dev [{ci1[0]:.2f} <-- {np.std(data):.2f} -->{ci1[1]:.2f}]") 
plt.show()