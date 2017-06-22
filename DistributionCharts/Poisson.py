import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

rv1 = poisson(1)
x = np.arange(0,15)
y1 = rv1.pmf(x)

plt.figure(figsize = (10,4))
plt.subplot(121)
plt.bar(x,y1, width=0.4)
plt.title("lamda = 1")
plt.xlabel("RV")
plt.ylabel("PMF")

plt.subplot(122)
rv2 = poisson(5)
y2 = rv2.pmf(x)

plt.bar(x,y2, width=0.4)
plt.title("lamda = 5")
plt.xlabel("RV")
plt.ylabel("PMF")

plt.show()