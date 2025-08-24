from PFM import util as myutil
import numpy as np
from sympy import *
from sympy.physics.wigner import gaunt

kd = KroneckerDelta


# Heaviside function
def hv(x):
    if x > 0:
        return 1
    else:
        return 0


# Unitary matrix that transforms complex sh to real sh
# See Eq. 12.
def U(l, m, mu):
    t1 = kd(m, 0) * kd(mu, 0)
    t2 = hv(mu) * kd(m, mu)
    t3 = hv(-mu) * I * ((-1) ** np.abs(m)) * kd(m, mu)
    t4 = hv(-mu) * (-I) * kd(m, -mu)
    t5 = hv(mu) * ((-1) ** np.abs(m)) * kd(m, -mu)
    return t1 + ((t2 + t3 + t4 + t5) / sqrt(2))


# Real gaunt coefficients
# See Eqs. 26. The sympy gaunt function does not use a complex conjugate.
# This sum could be truncated using selection rules, but this is fairly quick.
def Rgaunt(l1, l2, l3, m1, m2, m3, evaluate=True):
    result = 0
    for m1p in range(-l1, l1 + 1):
        U1 = U(l1, m1p, m1)
        for m2p in range(-l2, l2 + 1):
            U2 = U(l2, m2p, m2)
            for m3p in range(-l3, l3 + 1):
                U3 = U(l3, m3p, m3)
                result += U1 * U2 * U3 * gaunt(l1, l2, l3, m1p, m2p, m3p)
    if evaluate:
        return result.evalf()
    else:
        return result


G = np.zeros((6, 3, 3), dtype=np.complex64)
for lm in range(6):
    for i in range(3):
        for j in range(3):
            l, m = myutil.j2lm(lm)
            G[lm, i, j] = np.complex64(Rgaunt(l, 1, 1, m, i - 1, j - 1))
np.save('gaunt_633.npy', G)

# G = np.zeros((15, 15, 15), dtype=np.complex64)
# for index, g in np.ndenumerate(G):
#     l1, m1 = myutil.j2lm(index[0])
#     l2, m2 = myutil.j2lm(index[1])
#     l3, m3 = myutil.j2lm(index[2])
#     G[index] = np.complex64(gaunt(l1, l2, l3, m1, m2, m3))
# np.save('gaunt_l4', G)
