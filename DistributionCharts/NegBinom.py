from scipy.stats import nbinom
import numpy as np
import matplotlib.pyplot  as plt

def drawPmf(rv, length):
    x = np.arange(0, length, 1)
    y = rv.pmf(x)

    plt.subplot(121)
    plt.bar(x, y, width=0.4)

    plt.title("negative binormal distribution")
    plt.xlabel("k")
    plt.ylabel("P (X=k)")

    assert(sum(y) >= 0.99 and sum(y) <=1)

# samples: number of sample points
def drawSample(rv, samples):
    # generate sample of the distributions
    sample = rv.rvs(size=samples)
    # print(sample)

    # count the occurance of each sample numbers
    bCnt = np.bincount(sample)

    # generate x-axis
    xs = np.arange(0, np.size(bCnt), 1)

    # draw the statistics on 122
    plt.subplot(122)
    plt.bar(xs, bCnt/(samples*1.0), width=0.4)

rv = nbinom(3, 0.65)

drawPmf(rv, 10)
drawSample(rv, 1000)
plt.show()
