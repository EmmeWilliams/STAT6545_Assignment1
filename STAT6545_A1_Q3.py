import numpy as np
from numpy.random import normal, uniform, binomial, exponential
from scipy.special import factorial
from scipy.special import comb
import matplotlib.pyplot as plt

def mix(x):
    #Evaluate the target 
    return 0.2/(np.sqrt(np.pi))*np.exp(-(x-1)**2) + 0.8/(np.sqrt(0.2*np.pi))*np.exp((-(x-2)**2)/0.2)

x = np.linspace(-5,5,1000)
plt.plot(x,mix(x))
plt.title("Mixture $\pi(x)$")
plt.xlabel("x")
plt.savefig('./Stats6545_A1_3a_1.pdf')

print(mix(2))
print(mix(2)*np.sqrt(2*np.pi))

def q_mix(x):
    #Evaluate the proposal q(x) for the mixture π(x).
    return (1/np.sqrt(2*np.pi))*np.exp(-(x-2)**2/2)

M = 2.65
plt.plot(x,mix(x),label='$\pi(x)$')
plt.plot(x,q_mix(x),label='$q(x)$')
plt.plot(x,M*q_mix(x),label='$Mq(x)$')
plt.title("$\pi(x)$ and $q(x)$")
plt.xlabel("x")
plt.legend()
plt.savefig('./Stats6545_A1_3a_2.pdf')

np.random.seed(4000)
sample = normal(2,1,100000)
unif = uniform(0,1,100000)
accepted = sample[unif <= mix(sample)/(M*q_mix(sample))]

plt.hist(accepted,bins=1000,density=True)
plt.plot(x,mix(x),label='π(x)')
plt.xlim(-5,5)
plt.title('Histogram of $1e5$ samples using the rejection method')
plt.legend()
plt.savefig('./Stats6545_A1_3a_3.pdf')

rate = 0
np.random.seed(100)
for i in range(10):
    sample = normal(2,1,100000)
    unif = uniform(0,1,100000)
    accepted = sample[unif <= mix(sample)/(M*q_mix(sample))]
    rate += len(accepted)
print(rate/1000000)

def expo(x,λ=1,a=0):
    #Evaluate the density of Exponential(λ) or the shifted version
    y = np.zeros_like(x)
    y[x >= a] = λ*np.exp(-λ*(x[x >= a]-a))
    return y

fig, ax = plt.subplots(2,figsize=(8,6))
λ = 1
a = 1
iM = np.exp(-λ*a)
x = np.linspace(0,5,1000)
ax[0].plot(x,np.exp(λ*a)*expo(x, λ, 0),label='$Mq(x)$')
ax[0].plot(x,expo(x, λ, a),label='$π(x)$')
ax[0].legend()
ax[0].set_title(f'Comparing $q(x)$ and $π(x)$\n\n a={a}, λ={λ}, 1/M={iM}')
λ = 1
a = 2
iM = np.exp(-λ*a)
ax[1].plot(x,np.exp(λ*a)*expo(x, λ, 0),label='$Mq(x)$')
ax[1].plot(x,expo(x, λ, a),label='$π(x)$')
ax[1].legend()
ax[1].set_title(f'a={a}, λ={λ}, 1/M={iM}')
fig.tight_layout()
plt.savefig('./Stats6545_A1_3b_1.pdf')

x = np.linspace(0,10,100)
plt.plot(x,np.exp(-1*x),label='λ = 1')
plt.plot(x,np.exp(-2*x),label='λ = 2')
plt.plot(x,np.exp(-3*x),label='λ = 3')
plt.legend()
plt.title('Acceptance rate')
plt.xlabel('a')
plt.ylabel('1/M')
plt.savefig('./Stats6545_A1_3b_2.pdf')

fig, ax = plt.subplots(2,figsize=(8,6))
np.random.seed(4000)
λ = 1
a = 1
M = np.exp(λ*a)
sample = exponential(λ,1000)
unif = uniform(0,1,1000)
accepted = sample[unif <= expo(sample, λ, a)/(M*expo(sample, λ, 0))]
ax[0].hist(accepted,bins=100,density=True)
ax[0].plot(x,expo(x,λ,a),label='π(x)')
ax[0].legend()
ax[0].set_title(f'Histogram with 1000 samples using rejection\n\n a={a}, λ={λ}, 1/M={1/M}')
λ = 1
a = 2
M = np.exp(λ*a)
sample = exponential(λ,1000)
unif = uniform(0,1,1000)
accepted = sample[unif <= expo(sample, λ, a)/(M*expo(sample, λ, 0))]
ax[1].hist(accepted,bins=100,density=True)
ax[1].plot(x,expo(x,λ,a),label='π(x)')
ax[1].legend()
ax[1].set_title(f'a={a}, λ={λ}, 1/M={1/M}')
fig.tight_layout()
plt.savefig('./Stats6545_A1_3b_3.pdf')
