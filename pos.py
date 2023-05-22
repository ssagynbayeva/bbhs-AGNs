import numpy as np
import matplotlib.pyplot as plt

xp1 = []
yp1 = []
zp1 = []

xp2 = []
yp2 = []
zp2 = []

mass = [1.5e-5,1.5e-5]

a_bin = 0.03

eccent = 0.5

def position(time,obj):

      period = 2.0 * np.pi * np.sqrt((a_bin * 0.5) ** 3 / (2 * 1.5e-5));

      E = compute_newton_raphson(10,2 * np.pi * time / period,eccent);
      # Real rb0 = a_bin*(1-eccent);
      # Real v_b = sqrt(2*mass[0]/a_bin);
      # Real wb0 = v_b*sqrt((1-eccent)/(1+eccent));

      # g1 = a_bin/rb0*(1-cos(E))-1;
      # g2 = 1/ome*(E-sin(E))-time;
      # rb = g1*rb0+g2*wb0

      if obj==0:
        # xp1.append(0.5*a_bin*(np.cos(E)-eccent)-1)
        # yp1.append(0.5*a_bin*np.sqrt(1-eccent*eccent)*np.sin(E))
        # zp1.append(0.0)

        E = compute_newton_raphson(10,2 * np.pi * time / period,eccent); 
        f = 2.0 * np.arctan2(np.sqrt(1.0 - eccent * eccent) * np.sin(E) / (1.0 + eccent * np.cos(E)), np.cos(E) / (1.0 + eccent));
        xp1.append(0.5*a_bin * (np.cos(f) - eccent) - 1)
        yp1.append(0.5*a_bin * np.sqrt(1 - eccent * eccent) * np.sin(f))
        zp1.append(0.0)

        return xp1,yp1,zp1

      if obj==1:
        E = compute_newton_raphson(10,2 * np.pi * time / period ,eccent); 
        f = 2.0 * np.arctan2(np.sqrt(1.0 - eccent * eccent) * np.sin(E) / (1.0 + eccent * np.cos(E)), np.cos(E) / (1.0 + eccent));
        xp2.append(-0.5*a_bin * (np.cos(f) - eccent) - 1.)
        yp2.append(-0.5*a_bin * np.sqrt(1 - eccent * eccent) * np.sin(f))
        zp2.append(0.0)

        # xp2.append(-0.5*a_bin*(np.cos(E)-eccent)-1.03)
        # yp2.append(-0.5*a_bin*np.sqrt(1-eccent*eccent)*np.sin(E))
        # zp2.append(0.0)

        return xp2,yp2,zp2
      

   


def compute_newton_raphson(N_it, M, e):

    k = 0.85;

    this_ell = M;
    old_E = this_ell + k*e;

    # Define initial estimate
    if((np.sin(this_ell))<0):
        old_E = this_ell - k*e;
    else: 
        old_E = this_ell + k*e;

    # Perform Newton-Raphson estimate
    for j in range(N_it):

        f_E = old_E - e*np.sin(old_E)-this_ell;
        fP_E = 1. - e*np.cos(old_E);

        old_E = old_E - f_E/fP_E;
        new_E = old_E;


    # E = M

    # for _ in range(N_it):
    #     dE = -(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    #     E  = E+dE
    #     if abs(dE)<1e-7: break;

    # return E;
    return new_E

for i in range(1000):
    xp1,yp1,zp1 = position(i,obj=0)
    xp2,yp2,zp2 = position(i,obj=1)

plt.plot(xp1,yp1,'k.')
plt.plot(xp2,yp2,'r.')
plt.show()

print(np.array(xp1)-np.array(xp2))
