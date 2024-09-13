# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:52:42 2024

@author: Rohan Bapat
"""

import SimulateDemand
for i in range(10):
    for j in range(1,6):
        SimulateDemand.main(i, 'ccedisruption', 'continuous', 0, j)

    for j in range(5):
        SimulateDemand.main(i, 'replenishmentdisruption', 'continuous', j*0.05, 0)


#    SimulateDemand.main(i, 'continuous')
    