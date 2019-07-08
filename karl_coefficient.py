#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:04:14 2019

@author: neetika
"""


import math


physics_scores = [15,12,8, 8, 7, 7, 7, 6, 5, 3]
history_scores = [10,25,17,11,13,17,20,13,9,15]

N = len(physics_scores)
sum_xy = 0
xy = []
karl_coef = 0.0

for i in range(N):
    xy.append(physics_scores[i]*history_scores[i])

sum_xy = sum(xy)

sum_x = sum(physics_scores)
sum_y = sum(history_scores)

x_square = [i*i for i in physics_scores]
y_square = [i*i for i in history_scores]

sum_x_square = sum(x_square)
sum_y_square = sum(y_square)

square_sum_x = math.pow(sum_x, 2)
square_sum_y = math.pow(sum_y, 2)

numerator = N * sum_xy - sum_x * sum_y
coef_x = N * sum_x_square - square_sum_x
coef_y = N * sum_y_square - square_sum_y
denomenator = math.sqrt(coef_x) * math.sqrt(coef_y)

if denomenator != 0:
    karl_coef = numerator/denomenator

print('%.3f' % karl_coef)
