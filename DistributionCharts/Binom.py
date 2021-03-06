from scipy.stats import binom
import numpy as np
import matplotlib.pyplot  as plt

# length: the length of x-axis
def drawPmf(rv, length):
    x = np.arange(0, length, 1)
    y = rv.pmf(x)

    plt.subplot(121)
    plt.bar(x, y, width=0.4)

    plt.title("binomial distribution")
    plt.xlabel("RV")
    plt.ylabel("P(X=x)")
    assert( sum(y) >= 0.99)

# samples: number of sample points
def drawSample(rv, samples):
    # generate sample of the distributions
    sample = rv.rvs(size=samples)
    # print(sample)

    # count the occurance of each sample numbers
    bCnt = np.bincount(sample)
    xs = np.arange(0, np.size(bCnt), 1)

    # draw the statistics on 122
    plt.subplot(122)
    plt.bar(xs, bCnt/(samples*1.0), width=0.4)

rv = binom(10, 0.7)

drawPmf(rv, 12)
drawSample(rv, 10000)
plt.show()