import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# Data from the table (converted to hours)
methods = {
    "BitTorrent-u": {"mu": 8.36, "sigma": 5.16, "min": 2.21, "max": 14.39},
    "IPFS-u": {"mu": 10.68, "sigma": 9.54, "min": 1.80, "max": 24.62},
    "Rsync-u": {"mu": 4.84, "sigma": 1.39, "min": 3.10, "max": 6.10},
    "Girder-c": {"mu": 2.85, "sigma": 2.31, "min": 1.05, "max": 6.24},
    "HF-c": {"mu": 0.14, "sigma": 0.03, "min": 0.11, "max": 0.18},
    "Rsync-c": {"mu": 1.10, "sigma": 0.03, "min": 1.07, "max": 1.13},
}


# Function to sample from a truncated Gaussian
def sample_truncated_gaussian(mu, sigma, a, b, size=10000):
    a_norm = (a - mu) / sigma
    b_norm = (b - mu) / sigma
    return truncnorm.rvs(a_norm, b_norm, loc=mu, scale=sigma, size=size)

# Generate samples for each method
samples = {}
for method, params in methods.items():
    samples[method] = sample_truncated_gaussian(
        params["mu"], params["sigma"], params["min"], params["max"]
    )

# Compute pairwise expected speedups (ratio of times = inverse of speed ratios)
speedups = {}
methods_list = list(methods.keys())
for i in range(len(methods_list)):
    for j in range(len(methods_list)):
        method1, method2 = methods_list[i], methods_list[j]
        ratio = samples[method2] / samples[method1]  # time2 / time1 = 1/speedup
        speedup = np.mean(1 / ratio)  # Convert to speedup (time1/time2)
        pair = method1, method2
        speedups[pair] = speedup


# Display results sorted by speedup
print("Expected Speedup (X times faster):")
for pair, speedup in sorted(speedups.items(), key=lambda x: x[1], reverse=True):
    method1, method2 = pair
    print(f"{method1} → {method2}: {speedup:.2f}x")

piv = pd.DataFrame([(k[0], k[1], v) for k, v in speedups.items()]).pivot(index=0, columns=1, values=2)
print(piv.to_string())
