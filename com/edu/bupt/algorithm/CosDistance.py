__author__ = 'Administrator'

#
# this function is mainly used to measure the relationship between A and B
# for example, movie A and B will be scored by User C and D
# C is [1, 2] and D is [4, 5]
# if we use cos distance to measure the relationship between C and D, we found
# that C is closed to D because the distance is about 0.98, but if we do some
# pre-processing like some mean scores , A mean score is (1+4)/2=2.5 and B
# mean score is (2+5)/2=3.5, so C is [1-2.5, 2-3.5] and D is [4-2.5, 5-3.5]

from math import sqrt
import numpy as np


# directly cos distance
def cos_distance(user1, user2):
    cos_dis = np.dot(user1, user2)/(sqrt(np.dot(user1, user1))*sqrt(np.dot(user2, user2)))
    return cos_dis


user_c = np.array([1, 2])
user_d = np.array([4, 5])
print cos_distance(user_c, user_d)

# print cos_dis

user_mean = (user_c + user_d)/2
user_c1 = user_c-user_mean
user_d1 = user_d-user_mean
print cos_distance(user_c1, user_d1)
