import numpy as np
import matplotlib.pyplot as plt

Swap2 = np.array([
  [1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1]
])

Swap02 = np.eye(16)
Swap02[[2, 8]] = Swap02[[8, 2]]
Swap02[[3, 9]] = Swap02[[9, 3]]
Swap02[[6, 12]] = Swap02[[12, 6]]
Swap02[[7, 13]] = Swap02[[13, 7]]

# The linear entropy of an operator
def entropy(U):
  Udag = U.conjugate().T
  return 1 - (1/16)*np.trace(
    np.kron(U, U) @ Swap02 @ np.kron(Udag, Udag) @ Swap02
  )

eS2 = entropy(Swap2)
def epower(U):
  return (4/9)*(entropy(U) + entropy(U @ Swap2) - eS2)

#eS02S13 = entropy(Swap(0, 2) @ Swap(1, 3))
#def epoweranc(U):
#  return (16/25)*(entropy(U) + entropy(U @ Swap(0, 2) @ Swap(1, 3)) - eS02S13)

def Utheta(theta):
  U = np.eye(4, dtype=complex)
  U[3][3] *= np.exp(1j*theta)

  return U

Omega = 1e-2
Vct = 1

def _oexp(U, n):
  EU = np.zeros(U.shape, dtype=complex)
  
  Up = np.eye(U.shape[0], dtype=complex)
  for i in range(n):
    EU += Up/np.math.factorial(i)
    Up = U @ Up

  return EU

def oexp(U, n=20, lognsteps=15):
  nsteps = 2**lognsteps
  EU = _oexp(U/nsteps, n)
  for i in range(lognsteps):
    EU = EU @ EU

  return EU

Hrrrrct = np.zeros((9, 9), dtype=complex)
Hrrrrct[8, 8] = np.real(Vct)
r1proj = np.outer([0, 0, 1], [0, 1, 0]).astype(complex)
proj11 = np.outer([0, 1, 0], [0, 1, 0]).astype(complex)
projrr = np.outer([0, 0, 1], [0, 0, 1]).astype(complex)
def Upulse(Delta, phi, tau):
  Hr1 = (Omega/2)*(np.exp(1j*phi)*r1proj + np.exp(-1j*phi)*r1proj.T) - Delta*proj11

  return oexp(-1j*(Hrrrrct + np.kron(Hr1, np.eye(3, dtype=complex)) + np.kron(np.eye(3, dtype=complex), Hr1))*tau)

def pull01(U):
  return U[[0, 1, 3, 4],:][:,[0, 1, 3, 4]]

def score01(U):
  eta1 = 0
  for i in [0,1,3,4]:
    for j in [0,1,3,4]:
      eta1 += np.abs(U[i,j])
  
  eta2 = 0
  for i in [2,5,6,7,8]:
    for j in [0,1,3,4]:
      eta2 += np.abs(U[i,j]) + np.abs(U[j,i])

  return eta1 / (eta1 + eta2)

maxloss = 0
def loss(U, alpha=10, meta=None):
  global maxloss
  power = np.real(epower(pull01(U)))
  score = score01(U)
  loss = power*np.exp(alpha*(score - 1))

  if loss > maxloss:
    print("Power: %f  Score: %f  Loss: %f  Meta: %s" % (power, score, loss, meta))
    maxloss = loss
  return loss

def lossp(pulses, **kwargs):
  U = np.eye(9)
  for p in pulses:
    U = Upulse(p[0], p[1], p[2]) @ U

  return loss(U, **kwargs)

#Delta = np.linspace(0.4, 0.6, 100)*Omega
#phi = np.linspace(0, 2*np.pi, 40)
Delta = np.linspace(0.526, 0.53, 40)*Omega
phi = np.linspace(0, 2*np.pi, 40)
tau = np.linspace(83, 83.5, 1000)/Omega
#tau = 2.7328*np.pi/(2*Omega)

#pulse = Upulse(0.377*Omega, 3.90242, 2.7328*np.pi/(2*Omega)) @ Upulse(0.377*Omega, 0, 2.7328*np.pi/(2*Omega))
#print(score01(pulse))
#print(epower(pull01(pulse)))

print(entropy(np.array([
  [1,0,0,0],
  [0,1,0,0],
  [0,0,0,1],
  [0,0,1,0]
])))

#print(lossp([
#  [0.377*Omega, 0, 2.7328*np.pi/(2*Omega)],
#  [0.377*Omega, 3.90242, 2.7328*np.pi/(2*Omega)]
#]))
"""
losses = np.array([
  [ lossp([[d1, 0, tau1]]), d1, tau1 ] for d1 in Delta for tau1 in tau
])
best = losses[np.argmax(losses[:,0])]
print(best[0], best[1]/Omega, best[2]*Omega/np.pi)
"""

print(pull01(Upulse(0.0052833333333333335, 0, 8333.587114098655)))
losses = np.array([
  max([ lossp([[d1, 0, tau1]], meta=(d1, tau1)) for d1 in Delta]) for tau1 in tau
])
plt.plot(tau*Omega/np.pi, losses)
plt.xlabel("tau*Omega/pi")
plt.ylabel("1-L(U)")
plt.show()
