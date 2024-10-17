import numpy as np
from numpy.random import normal, uniform, binomial, exponential
from scipy.special import factorial
from scipy.special import comb
import matplotlib.pyplot as plt

def binom_cdf(k):
    #Recursively calculate the CDF for a binomial distribution
    i = 0
    cdf = binom(i)
    while i < k:
        i += 1
        cdf += binom(i)
    return cdf
def binom(k):
    #Evaluate the PDF for Binomial(10,1/3)
    return comb(10,k)*(1/3)**k*(2/3)**(10-k)

np.random.seed(2000)
x = np.linspace(0,10,11)
probabilities = np.array([binom_cdf(k) for k in x]) #all possible outcomes
uni = uniform(0,1,1000)
outcome = np.zeros_like(uni)
for i in range(len(uni)):
    ind = 0
    while (probabilities[ind] < uni[i]) and (ind < 10):
        ind += 1
    outcome[i] = ind #select the appropriate interval & outcome for each sample

x2 = np.linspace(0,10,100)
plt.figure(figsize=(6,6))
plt.hist(outcome,density=True,align='left',bins=[0,1,2,3,4,5,6,7,8,9,10,11])
plt.plot(x2,binom(x2),label='Binomial(10,1/3)')
plt.title('Histogram of 1000 samples from Binomial(10,1/3) \nusing the inversion method')
plt.legend()
plt.savefig('./Stats6545_A1_1.pdf')

def bernoulli_cdf(k):
    #Evaluate the CDF for Bernoulli(1/3)
    return (0 if k < 2/3 else 1)
np.random.seed(1000)
outcome2 = np.zeros(1000)
for i in range(1000):
    uni = uniform(0,1,10)
    outcome = np.array([bernoulli_cdf(k) for k in uni]) #inversion
    outcome2[i] = np.sum(outcome) #transformation
plt.figure(figsize=(6,6))
plt.hist(outcome2,density=True,align='left',bins=[0,1,2,3,4,5,6,7,8,9,10,11])
x2 = np.linspace(0,10,100)
plt.plot(x2,binom(x2),label='Binomial(10,1/3)')
plt.title('Histogram of 1000 samples from Binomial(10,1/3) \n using the combination method')
plt.legend()
plt.savefig('./Stats6545_A1_1b.pdf')

np.random.seed(1000)
x = np.linspace(0,10,11)
probabilities = np.array([binom_cdf(k) for k in x]) #all possible outcomes
uni = uniform(0,1,100)
outcome = np.zeros_like(uni)
for i in range(len(uni)):
    ind = 0
    while (probabilities[ind] < uni[i]) and (ind < 10):
        ind += 1
    outcome[i] = ind #select the appropriate interval & outcome for each sample

mu = np.mean(outcome)
v = np.var(outcome)
sigma = np.sqrt(v/100)
print(mu)
print(v)
print(sigma)

print(mu - 1.96*sigma, mu + 1.96*sigma)
print(mu - 2.576*sigma, mu + 2.576*sigma)
