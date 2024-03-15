import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
incomes = np.random.normal(27000,15000,10000)
mean1 =np.mean(incomes)

plt.hist(incomes,50)
plt.show()

median1=np.median(incomes)
print(mean1, median1)

incomes = np.append(incomes, [1000000000])
median2 = np.median(incomes)
mean2 = np.mean(incomes)
print(mean2, median2)

ages = np.random.randint(18,high=90,size =500)
print(ages)

mode1 = stats.mode(ages)
print(mode1)