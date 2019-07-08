#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:33:39 2019

@author: neetika
"""

# Enter your code here. Read input from STDIN. Print output to STDOUT
import statistics
import math

physics_scores = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
 
history_scores = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]

physics_mean = statistics.mean(physics_scores)
history_mean = statistics.mean(history_scores)

deviation_from_mean_x = [(x - physics_mean) for x in physics_scores]
deviation_from_mean_y = [(x - history_mean) for x in history_scores]

product = []
j = 0
for i in range(len(deviation_from_mean_x)):
    product.append(deviation_from_mean_x[j]*deviation_from_mean_y[j])
    j = j + 1

product_sum = sum(product)
mean_square_x = [math.pow(x, 2) for x in deviation_from_mean_x]
mean_square_x_sum = sum(mean_square_x)

slope = product_sum/mean_square_x_sum

print(round(slope, 3))