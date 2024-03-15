import numpy as np
import matplotlib.pyplot as plt
incomes = np.random.normal(200.0, 30.0, 10000)

plt.hist(incomes, 50)
plt.show()

std = incomes.std()
variance = incomes.var()

print(std,variance)
