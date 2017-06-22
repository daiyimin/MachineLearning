from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

def drawPdf(rv, xLimit, dots):
    x = np.linspace(0, xLimit, dots)

    plt.subplot(121)
    plt.plot(x, rv.pdf(x))
    plt.xlim([0, xLimit])
    plt.title("exponential distribution")
    plt.xlabel("RV")
    plt.ylabel("f(x)")

# samples: number of sample points
def drawSample(rv, samples, xLimit):
    # generate sample points, the bigger size the accurate result
    sample = rv.rvs(size=samples)

    x = np.arange(0, xLimit, 1)
    y = np.zeros(xLimit)
    # iterate through x-axis
    for i in x:
        # filter out the sample points that drops around i, the x coordinate scope is (i < x <= i+1)
        sampleAroundI = sample[(sample > i) & (sample <= i+1)]
        # (the probability of event happens around i) = (number of filtered sample points) / (total sample points)
        # save (the probability of event happens around i) in y[i]
        y[i] = sampleAroundI.size * 1.0 / samples

    plt.subplot(122)
    plt.plot(x, y)

rv = expon(scale = 5)

drawPdf(rv, 20, 100)
drawSample(rv, 20000, 21)

plt.show()