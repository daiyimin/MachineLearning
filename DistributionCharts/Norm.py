from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

def drawPdf(rv, xLimit, dots):
    x = np.linspace(-xLimit, xLimit, dots)

    plt.subplot(121)
    plt.plot(x, rv.pdf(x))
    plt.xlim([-xLimit, xLimit])
    plt.title("exponential distribution")
    plt.xlabel("RV")
    plt.ylabel("f(x)")

# samples: number of sample points
def drawSample(rv, samples, xLimit):
    # generate sample points, the bigger size the accurate result
    sample = rv.rvs(size=samples)

    x = np.arange(-xLimit, xLimit + 1, 1)
    y = np.zeros(2*xLimit + 1)
    idx = 0
    # iterate through x-axis
    for i in x:
        # filter out the sample points that drops around i, the x coordinate scope is (i < x <= i+1)
        sampleAroundI = sample[(sample > i) & (sample <= i+1)]
        # (the probability of event happens around i) = (number of filtered sample points) / (total sample points)
        # save (the probability of event happens around i) in y[i]
        y[idx] = sampleAroundI.size * 1.0 / samples
        idx += 1

    plt.subplot(122)
    plt.plot(x, y)

rv1 = norm(loc=0, scale = 1)
rv2 = norm(loc=2, scale = 1)
rv3 = norm(loc=0, scale = 2)



drawPdf(rv1, 5, 200)
drawSample(rv1, 50000, 5)

plt.show()