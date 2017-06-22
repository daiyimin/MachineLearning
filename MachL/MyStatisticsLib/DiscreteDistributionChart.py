import numpy as np

# draw Probability Mass Function of discrete random variable
# rv, a discrete random variable
# length, number of bars to be drawn
# plt, plot
def drawPmf(rv, length, plt):
    x = np.arange(0, length, 1)
    y = rv.pmf(x)

    plt.bar(x, y, width=0.4)

# samples: number of sample points
def drawSample(rv, samples, plt):
    # generate sample of the distributions
    sample = rv.rvs(size=samples)

    # count the occurance of each sample numbers
    # the count is the number of bars to be drawn
    bCnt = np.bincount(sample)
    xs = np.arange(0, np.size(bCnt), 1)

    # draw the statistics
    plt.bar(xs, bCnt/float(samples), width=0.4)