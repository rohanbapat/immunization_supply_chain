# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:52:42 2024

@author: Rohan Bapat
"""

import SimulateDemand
for i in range(5):
    SimulateDemand.main(i, 'periodic')
    SimulateDemand.main(i, 'continuous')
    