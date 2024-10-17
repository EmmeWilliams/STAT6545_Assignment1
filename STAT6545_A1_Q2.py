import numpy as np
from numpy.random import normal, uniform, binomial, exponential
from scipy.special import factorial
from scipy.special import comb
import matplotlib.pyplot as plt

def poisson(k,λ=1):
    #Evaluate the Poisson(k) distribution
    return (λ**k)*np.exp(-λ)/factorial(k)

np.random.seed(2000)
n = 1000
counts = np.zeros(n)
k = np.linspace(0,6,7)
for i in range(n):
    sum = 0
    count = 0
    while sum < 1: #transformation
        sum += -np.log(1-uniform(0,1,1)) #inversion
        count += 1
    counts[i] = count-1
plt.hist(counts,density=True,align='left',bins=10)
plt.plot(k,2*poisson(k,1),label='Poisson(t)')
plt.xlabel('t')
plt.title('Histogram with 1000 samples for Poisson(t)')
plt.legend()
plt.savefig('./Stats6545_A1_2.pdf')

np.random.seed(5000)
for n in [10,100,1000,10000]:
    counts = np.zeros(n)
    k = np.linspace(0,6,7)
    for i in range(n):
        sum = 0
        count = 0
        while sum < 1: #transformation
            sum += -np.log(1-uniform(0,1,1)) #inversion
            count += 1
        counts[i] = count-1
    m = np.mean(counts)
    std = np.std(counts,ddof=1)/np.sqrt(n)
    low = m - 1.96*std
    high = m + 1.96*std
    print(f'For n={n} samples, the mean is {m}, the standard error is {round(std,4)}, and the 95% confidence bounds are ({round(low,4)},{round(high,4)})')


