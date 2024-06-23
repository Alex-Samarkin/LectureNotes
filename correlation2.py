import numpy as np

def spearman_corr(a, b):
    """
    Calculates the Spearman correlation coefficient between two one-dimensional arrays.

    Args:
    a: A numpy array.
    b: A numpy array of the same length as a.

    Returns:
    The Spearman correlation coefficient between a and b.
    """
    # Rank the data points
    ranked_a = a.argsort() + 1
    ranked_b = b.argsort() + 1

    # Calculate the difference in ranks
    d = ranked_a - ranked_b

    # Calculate the squared difference in ranks
    d_sq = d**2

    # Calculate the number of data points
    n = len(a)

    # Calculate the Spearman correlation coefficient
    rho = 1 - 6 * np.sum(d_sq) / (n * (n**2 - 1))

    return rho

# Test the function with example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
z = np.array([-1, 2, -3, 4, -5]) 

print(spearman_corr(x, y))  # Expected output: -1.0
print(spearman_corr(x, x))  # Expected output: 1.0
print(spearman_corr(y, y))  # Expected output: 1.0
print(spearman_corr(x, z))  # Expected output: ??

# generate n ranked data from 0 to 9 and shuffle it
def generate_ranked_data(n, max_val=9):
    data = np.arange(max_val + 1)
    np.random.shuffle(data)
    data = data[:n]
    return data
x = generate_ranked_data(1000)
y = generate_ranked_data(1000)
z =  x+y

# print the spearman correlation coefficient between x and y using spearmanr from scipy.stats
from scipy.stats import spearmanr
print(spearmanr(x, y))
print(spearmanr(x, x))
print(spearmanr(x, z))
print(spearmanr(y, z))

print(np.corrcoef([x, y, z]))

