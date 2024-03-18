from scipy.stats import expon 
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,10,0.001)
plt.plot(x, expon.pdf(x))
plt.show()