import numpy as np
import matplotlib.pyplot as plt

def norm(b):
    a = 1/b
    eth1 = 1 - a/2 + (1/2)*np.sqrt(a*(a-4), dtype=complex)
    eth2 = 1 - a/2 - (1/2)*np.sqrt(a*(a-4), dtype=complex)

    return (np.minimum((eth1*np.conj(eth1)).astype(float), 2), np.minimum((eth2*np.conj(eth2)).astype(float), 2))

N = 1000
a = np.arange(1, N, 1)/N
norms = norm(a)
plt.plot(a, norms[0])
plt.plot(a, norms[1])
plt.show()
