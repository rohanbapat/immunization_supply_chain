# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:52:42 2024

@author: Rohan Bapat
"""

import pandas as pd

import SimulateDemand

from itertools import product

from pathlib import Path

path_data = Path.cwd().parent / 'data'

#scen_name = ['baseline', 'ccedisruption', 'replenishmentdisruption'] # Its possible to specify both cce disruption and replenishment disruption, but avoid
#ss_mode = ['periodic', 'continuous']
#ss_replenishment_source = ['phc', 'dvs']
#phc_replenishment_disrupt = 0
#annual_cce_disruptions = 0

median_delay_df = pd.DataFrame()
delay_greater_30_df = pd.DataFrame()
frac_vac_df = pd.DataFrame()



scen_params_setting = {'scenario_name': ['ccedisruption'],
                'ss_mode': ['periodic-2device'],
#                'ss_mode': ['periodic', 'continuous'],
#                'ss_replenishment_source': ['phc', 'dvs'],
                'ss_replenishment_source': ['phc'],
#                'replenishmentdisruption': [0.05, 0.1, 0.15, 0.2],
#                'replenishmentdisruption': [0.1, 0.2],
                'replenishmentdisruption': [0],
                'ccedisruption': [0],
#                'ccedisruption': [1/365, 2/365, 3/365, 4/365]
                }

keys, values = zip(*scen_params_setting.items())
scen_combos = [dict(zip(keys, p)) for p in product(*values)]
ct = 0

for scen_params in scen_combos:
    if (scen_params['ss_mode'] == 'periodic') & (scen_params['ss_replenishment_source']=='dvs'):
        continue
    else:
        for i in range(100):
        
            median_delay, delay_greater_30, frac_vac = SimulateDemand.main(i, scen_params['scenario_name'], scen_params['ss_mode'], scen_params['ss_replenishment_source'], scen_params['replenishmentdisruption'], scen_params['ccedisruption'])
            median_delay_scen_param = {**scen_params, **median_delay}
            delay_greater_30_scen_param = {**scen_params, **delay_greater_30}
            frac_vac_scen_param = {**scen_params, **frac_vac}
            
            median_delay_df = median_delay_df.append(pd.DataFrame(median_delay_scen_param, index=[ct]))
            delay_greater_30_df = delay_greater_30_df.append(pd.DataFrame(delay_greater_30_scen_param, index=[ct]))
            frac_vac_df = frac_vac_df.append(pd.DataFrame(frac_vac_scen_param, index=[ct]))
        
        ct+=1

median_delay_df.to_csv(path_data / '07 median_delay_2device_df.csv', index = False)
delay_greater_30_df.to_csv(path_data / '07 delay_greater_30_2device_df.csv', index = False)
frac_vac_df.to_csv(path_data / '07 frac_vac_2device_df.csv', index = False)