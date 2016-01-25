__author__ = 'jiangdon'


import numpy as np
import matplotlib.pylab as plt


N = 10
# generate N number (0, 1)
x_list = np.linspace(0, 1, N)
# generate N point with Normal gauss(0, 0.2)
y_list = np.sin(2*np.pi*x_list)+np.random.normal(0, 0.2, x_list.size)

# x and y is used for generating the sin function
x = np.linspace(0, 1, 1000)
y = np.sin(2*np.pi*x)

# plot the sin function and fixed N point
plt.plot(x, y, 'r-')
plt.plot(x_list, y_list, 'go')
plt.xlim(0, 1)
plt.ylim(-1.5, 1.5)
plt.show()