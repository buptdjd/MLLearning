__author__ = 'jiangdon'

import random


def markov_pi(n_trails, delta):
    x, y = 1, 1
    n_hits = 0
    for i in range(n_trails):
        delta_x = random.uniform(-delta, delta)
        delta_y = random.uniform(-delta, delta)
        if abs(x+delta_x) < 1.0 and abs(y+delta_y) < 1.0:
            x, y = x+delta_x, y+delta_y
        if x**2+y**2 < 1.0:
            n_hits += 1
    return n_hits


def caculate_pi(n_trails):
    return 4*float(markov_pi(n_trails, 0.1))/float(n_trails)