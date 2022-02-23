import numpy as np
import matplotlib.pyplot as plt
delta= np.array([[0.04,0.0192,-0.0024],[0.0192,0.0256,-0.00096],[-0.0024,-0.00096,0.0036]])
delta1=np.linalg.inv(delta)
un=np.array([[1],[1],[1]])
unt=np.transpose(un)
R=np.array([[0.12],[0.1],[0.06]])
Rt=np.transpose(R)
r_list=np.linspace(0,1,100)

A=np.dot(np.dot(unt,delta1),R)
B=np.dot(np.dot(unt,delta1),un)
C=np.dot(np.dot(Rt,delta1),R)
risque=[]
x1,x2,x3=[],[],[]
for r in r_list:
  lamb=(C-A*r)/(A**2-B*C)
  mu=(A-r*B)/(A**2-B*C)
  x=np.dot(delta1,(mu*R-lamb*un))
  x1.append(x[0])
  x2.append(x[1])
  x3.append(x[2])
  risque.append(np.dot(np.dot(np.transpose(x),delta),x)[0][0])

plt.plot(risque,r_list)
plt.show()

plt.plot(r_list,x3)
plt.show()