# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:55:29 2025

@author: Rohan Bapat
"""

import SimulateDemand

import tssq_model_simulation

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

n_iter = 100

path_data = Path.cwd().parent / 'data'

#--------------------------------------------------------
for i in range(n_iter):
    SimulateDemand.main(i, 'baseline', 'periodic', 'phc', 0, 0)

choice_model_delay_hist = np.genfromtxt(path_data / '07 out_total_delay_phc_baseline_periodic.csv', delimiter=',')

l1 = np.percentile(choice_model_delay_hist,2.5, axis = 0)
m1 = np.percentile(choice_model_delay_hist,50, axis = 0)
u1 = np.percentile(choice_model_delay_hist,97.5, axis = 0)

plt.errorbar(list(range(0,100,5)), m1, yerr=[m1-l1,u1-m1], fmt='o', elinewidth = 0.5, capsize = 4, label = 'abc')
plt.bar(list(range(0,100,5)),m1, width = 4, alpha = 0.5)
#plt.show()
plt.savefig(r'C:\Users\Rohan Bapat\Documents\Projects\Immunization Supply Chain\Resources\choice_delay_histogram_with_ci.png')


#---------------------------------------------------------



hist_array = np.zeros([n_iter*2,20])
s_gaps_array = np.zeros([n_iter, 2])
d_gaps_array = np.zeros([n_iter, 2])

for i in range(n_iter):
    hist_data, s_gaps, d_gaps = tssq_model_simulation.main(i)

    hist_array[2*i,:] = hist_data[0][0]
    hist_array[2*i+1,:] = hist_data[1][0]
    
    s_gaps_array[i,:] = s_gaps
    d_gaps_array[i,:] = d_gaps
    

l = np.percentile(hist_array,2.5, axis = 0)
m = np.percentile(hist_array,50, axis = 0)
u = np.percentile(hist_array,97.5, axis = 0)
    
plt.errorbar(list(range(0,100,5)), m, yerr=[m-l,u-m], fmt='o', elinewidth = 0.5, capsize = 4, label = 'abc')
plt.bar(list(range(0,100,5)),m, width = 4, alpha = 0.5)
#plt.show()
plt.savefig(r'C:\Users\Rohan Bapat\Documents\Projects\Immunization Supply Chain\Resources\delay_histogram_with_ci.png')


l2 = np.percentile(s_gaps_array.flatten(),2.5)
m2= np.percentile(s_gaps_array.flatten(),50)
u2 = np.percentile(s_gaps_array.flatten(),97.5)
print(l2,m2,u2)


l3 = np.percentile(d_gaps_array.flatten(),2.5)
m3 = np.percentile(d_gaps_array.flatten(),50)
u3 = np.percentile(d_gaps_array.flatten(),97.5)
print(l3,m3,u3)