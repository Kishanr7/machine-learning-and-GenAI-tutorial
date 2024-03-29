from scipy.stats import poisson 
import matplotlib.pyplot as plt
import numpy as np

mu = 500
x = np.arange(400, 600, 0.5)
plt.plot(x, poisson.pmf(x, mu))
plt.show()