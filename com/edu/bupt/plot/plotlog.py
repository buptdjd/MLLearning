__author__ = 'jiangdon'

import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    x = [float(i)/100 for i in range(1, 300)]
    y = [math.log(i) for i in x]
    plt.plot(x, y, 'r-', linewidth=3, label="log curve")
    a = [x[20], x[175]]
    b = [y[20], y[175]]
    plt.plot(a, b, 'g-', linewidth=2)
    plt.plot(a, b, 'b*', markersize=15, )
    plt.legend(loc="upper left")
    plt.xlabel("x")
    plt.ylabel("log(x)")
    plt.grid()
    plt.show()
