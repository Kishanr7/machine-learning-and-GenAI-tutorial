import numpy as np
import matplotlib.pyplot as plt
incomes = np.random.normal(100.0 , 20.0 , 10000)
plt.hist(incomes, 50)
plt.show()

mean1 = np.mean(incomes)
median1 = np.median(incomes)
print(mean1, median1)