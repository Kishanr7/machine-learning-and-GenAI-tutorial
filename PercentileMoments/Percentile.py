import matplotlib.pyplot as plt
import numpy as np

vals = np.random.normal(0,0.5,10000)

plt.hist(vals,50)
plt.show()

p1 = np.percentile(vals,50)
p2 = np.percentile(vals,90)
p3 = np.percentile(vals,20)

print(p1,p2,p3)