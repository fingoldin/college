import numpy as np
import matplotlib.pyplot as plt

alpha = 0.005
L = 1
c = 3e8

n = np.arange(1, 8, 1)

k = n*np.pi/L
o = c*k*(1+alpha*k**2/2)

plt.plot(n, o*1e-9, label="Stiff string")
plt.plot(n, c*k*1e-9, label="Flexible string")
plt.legend()
plt.xlabel("n")
plt.ylabel("Angular frequency (GHz)")
plt.savefig("5b.pdf")
