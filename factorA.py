import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib

url = '''https://raw.githubusercontent.com/EducationalTestingService/factor_analyzer/main/tests/data/test02.csv'''
df_features = pd.read_csv(url)
print(df_features.head())

# Create factor analysis object and perform factor analysis 
fa = FactorAnalyzer(n_factors=3, rotation=None)
fa.fit(df_features)
# Check Eigenvalues
ev, _ = fa.get_eigenvalues()
print(ev)

# Create scree plot
import matplotlib.pyplot as plt
import seaborn as sns

# Create the scree plot
plt.scatter(range(1, len(ev) + 1), ev/sum(ev)*100)
plt.plot(range(1, len(ev) + 1), ev/sum(ev)*100)

# Add labels and title
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')

# Add gridlines for better visualization
plt.grid(True)
# Add horisonal line at 10
plt.axhline(y=10, color='red', linestyle='--')

# plot next on secondary y-axis for better visualization
ax2 = plt.twinx()
ax2.plot(range(1, len(ev) + 1), (ev/sum(ev)*100).cumsum(), 'o--', color='red')

# Show the plot
plt.show()

# Get the factor loadings matrix
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
fa.fit(df_features)
factor_loadings = fa.loadings_
# make pandas dataframe
factor_loadings = pd.DataFrame(factor_loadings, index=df_features.columns)
print(factor_loadings)

# plot every factor as a barplot
factor_loadings.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Factors')
plt.ylabel('Loading')
plt.title('Factor Loadings')
plt.show()

sns.heatmap(factor_loadings, annot=True, cmap='coolwarm')
plt.title('Factor Loadings')
plt.show()

# Get the factor scores
factor_scores = fa.transform(df_features)
factor_scores = pd.DataFrame(factor_scores, index=df_features.index)
print(factor_scores)
