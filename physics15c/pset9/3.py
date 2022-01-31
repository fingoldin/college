import numpy as np
import matplotlib.pyplot as plt

def E(kz, ot):
  return np.cos(kz)**2*np.cos(ot)**2+np.sin(kz)**2*np.sin(ot)**2

def S(kz, ot):
  return np.sin(2*kz)*np.sin(2*ot)

kz = np.linspace(0, np.pi, 1000)

plt.plot(kz, E(kz, 0), label="t=0")
plt.plot(kz, E(kz, np.pi/2), label="omega t=pi/2")
plt.plot(kz, E(kz, np.pi), label="omega t=pi")
plt.xlabel("kz")
plt.ylabel("Normalized energy density")
plt.legend()
plt.savefig("3a.pdf")

plt.figure()

plt.plot(kz, S(kz, np.pi/4), label="omega t=pi/4")
plt.plot(kz, S(kz, 3*np.pi/4), label="omega t=3pi/4")
plt.xlabel("kz")
plt.ylabel("Normalized z-component of Poynting vector")
plt.legend()
plt.savefig("3b.pdf")

