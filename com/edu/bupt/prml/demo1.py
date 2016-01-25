__author__ = 'jiangdon'


import numpy as np
import matplotlib.pylab as plt


N = 10
xlist = np.linspace(0, 1, N)
ylist = np.sin(2*np.pi*xlist)+np.random.normal(0, 0.2, xlist.size)
x = np.linspace(0, 1, 1000)
y = np.sin(2*np.pi*x)

plt.plot(x, y, 'r-')
plt.plot(xlist, ylist, 'go')
plt.xlim(0, 1)
plt.ylim(-1.5, 1.5)
plt.show()