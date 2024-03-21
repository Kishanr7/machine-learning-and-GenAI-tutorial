import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

vals = np.random.normal(10,15,10000)

plt.hist(vals,50)
plt.show()

mean = np.mean(vals)

var = np.var(vals)

skew = sp.skew(vals)

kurtosis = sp.kurtosis(vals)

print(mean, var, skew, kurtosis)