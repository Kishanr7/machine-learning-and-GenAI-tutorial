from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3,3,0.001)
plt.plot(x, norm.pdf(x))
plt.show()