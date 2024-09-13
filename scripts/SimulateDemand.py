
# coding: utf-8

# In[1]:

import numpy as np

import matplotlib.pyplot as plt

import random

import pandas as pd

from pathlib import Path

import time

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# In[7]:

def multinomial_logit(x):
    e0 = np.exp(x[0])
    e1 = np.exp(x[1])
    e2 = np.exp(x[2])
    den = e0+e1+e2
    return e0/den, e1/den, e2/den


# In[8]:

def identity_choice(x):
    argmx = np.argmax(x)
    x = [0] * len(x)
    x[argmx] = 1
    return x


# In[9]:

def create_p_matrix(i, attribute_matrix, beta_matrix, sigma_matrix, p_matrix):
    v_matrix = np.matmul(attribute_matrix, beta_matrix).T
    u_matrix = v_matrix + sigma_matrix[:,:,i]
    p_matrix[:,:,i] = np.apply_along_axis(multinomial_logit, 0, u_matrix) 
    return p_matrix


# In[10]:

def demand_estimation(i, p_matrix, supply_demand_combined, supply_demand_hc_level):
#    unserved_demand_today = supply_demand_combined['demand_unserved'][:i+1] * np.argmax(p_matrix[:,:i+1,i], axis = 0)
    unserved_demand_today = supply_demand_combined['demand_unserved'][:i+1]
    queued_demand_today = unserved_demand_today * supply_demand_combined['demand_queued'][:i+1]
    unqueued_demand_today = unserved_demand_today * (1 - supply_demand_combined['demand_queued'][:i+1])
    caregivers_choice = np.apply_along_axis(identity_choice, 0, p_matrix[:,:i+1,i])

    actual_phc = unqueued_demand_today * caregivers_choice[1]
    actual_session_site = unqueued_demand_today * caregivers_choice[2] + queued_demand_today
    
    supply_demand_hc_level['actual_demand']['phc'] = actual_phc
    supply_demand_hc_level['actual_demand']['ss'] = actual_session_site    
    return supply_demand_hc_level


# In[11]:

def starting_capacity_today_func_periodic(today, phc_attributes, supply_demand_combined, supply_demand_hc_level):
        
    if ((today%phc_attributes['phc_replenishment_frequency']==0) & (random.random() > phc_attributes['phc_replenishment_disruption'])) or (today == 0): 
        supply_demand_combined['starting_capacity'][today] = phc_attributes['max_stock_S']
    else:
        supply_demand_combined['starting_capacity'][today] = max(supply_demand_combined['starting_capacity'][today-1] - supply_demand_hc_level['demand_served']['phc'][today-1] - supply_demand_hc_level['demand_served']['ss'][today-1], 0)
    
    if today in phc_attributes['phc_cce_disruption']:
        supply_demand_combined['starting_capacity'][today] = 0
    
    return supply_demand_combined


def starting_capacity_today_func_continuous(today, phc_attributes, ss_attributes, supply_demand_hc_level):
        
    if ((today%phc_attributes['phc_replenishment_frequency']==0) & (random.random() > phc_attributes['phc_replenishment_disruption'])) or (today == 0): 
        # PHC Replenishment day & PHC replenishment not disrupted or Day 0
        supply_demand_hc_level['starting_capacity']['phc'][today] = phc_attributes['max_stock_S']
        supply_demand_hc_level['starting_capacity']['ss'][today] = max(supply_demand_hc_level['starting_capacity']['ss'][today-1] - supply_demand_hc_level['demand_served']['ss'][today-1],0)
    elif ((today%ss_attributes['ss_replenishment_frequency']==ss_attributes['session_start']) & (random.random() > ss_attributes['ss_replenishment_disruption']) & (ss_attributes['ss_replenishment_source']=='phc')) :
        # SS Replenishment day & SS Replenishment not disrupted and SS replenished from PHC
        supply_demand_hc_level['starting_capacity']['phc'][today] = max(supply_demand_hc_level['starting_capacity']['phc'][today-1] - supply_demand_hc_level['demand_served']['phc'][today-1] - ss_attributes['max_stock_S'], 0)
        supply_demand_hc_level['starting_capacity']['ss'][today] = min(ss_attributes['max_stock_S'], supply_demand_hc_level['starting_capacity']['phc'][today-1] - supply_demand_hc_level['demand_served']['phc'][today-1])
    elif ((today%ss_attributes['ss_replenishment_frequency']==ss_attributes['session_start']) & (random.random() > ss_attributes['ss_replenishment_disruption']) & (ss_attributes['ss_replenishment_source']=='dvs')) :
        # SS Replenishment day & SS Replenishment not disrupted and SS replenished from DVS
        supply_demand_hc_level['starting_capacity']['phc'][today] = max(supply_demand_hc_level['starting_capacity']['phc'][today-1] - supply_demand_hc_level['demand_served']['phc'][today-1], 0)
        supply_demand_hc_level['starting_capacity']['ss'][today] = ss_attributes['max_stock_S']
    else:
        supply_demand_hc_level['starting_capacity']['phc'][today] = max(supply_demand_hc_level['starting_capacity']['phc'][today-1] - supply_demand_hc_level['demand_served']['phc'][today-1],0)
        supply_demand_hc_level['starting_capacity']['ss'][today] = max(supply_demand_hc_level['starting_capacity']['ss'][today-1] - supply_demand_hc_level['demand_served']['ss'][today-1],0)
    
    if today in phc_attributes['phc_cce_disruption']:
        supply_demand_hc_level['starting_capacity']['phc'][today] = 0

    if today in ss_attributes['ss_cce_disruption']:
        supply_demand_hc_level['starting_capacity']['ss'][today] = 0
    
    return supply_demand_hc_level



# In[12]:

def supply_demand_periodic(today, session_frequency, sdc, sdhl, type_hc):
    
    if type_hc == 'phc':
        starting_capacity_today = max(sdc['starting_capacity'][today] - np.sum(sdhl['demand_served']['ss'][today]),0)
    else:
        starting_capacity_today = sdc['starting_capacity'][today]                                                                   

    if ((today+7)%session_frequency==0) or (type_hc == 'phc'):
        # Session day at session site or regular day at PHC
        if np.sum(sdhl['actual_demand'][type_hc]) <= starting_capacity_today:
            served_indices = np.nonzero(sdhl['actual_demand'][type_hc])[0]
            sdhl['demand_fulfilled_dates'][type_hc][served_indices] = today
            sdhl['demand_served'][type_hc][today] = np.sum(sdhl['actual_demand'][type_hc])
            sdc['demand_unserved'][served_indices] = 0
            sdc['demand_queued'][served_indices] = 0            
        else:
            queued_indices = np.nonzero(sdhl['actual_demand'][type_hc])[0]
            luckyones = random.sample(list(queued_indices), starting_capacity_today)
            unluckyones = queued_indices[~np.isin(queued_indices,luckyones)]
            sdhl['demand_fulfilled_dates'][type_hc][luckyones] = today   
            sdhl['demand_served'][type_hc][today] = sdc['starting_capacity'][today]     
            sdc['demand_unserved'][luckyones] = 0
            sdc['demand_queued'][luckyones] = 0
            sdc['demand_queued'][unluckyones] = 0
            sdhl['bad_experience'][type_hc][unluckyones] = 1
    else:
        # Not a session day at session site
        unserved_indices = np.nonzero(sdhl['actual_demand']['ss'])[0]
        sdc['demand_unserved'][unserved_indices] = 1
        sdc['demand_queued'][unserved_indices] = 1
        
    return sdc, sdhl


# In[12]:

def supply_demand_continuous(today, session_frequency, sdc, sdhl, type_hc):
    
    starting_capacity_today = sdhl['starting_capacity'][type_hc][today] 

    if np.sum(sdhl['actual_demand'][type_hc]) <= starting_capacity_today:
        served_indices = np.nonzero(sdhl['actual_demand'][type_hc])[0]
        sdhl['demand_fulfilled_dates'][type_hc][served_indices] = today
        sdhl['demand_served'][type_hc][today] = np.sum(sdhl['actual_demand'][type_hc])
        sdc['demand_unserved'][served_indices] = 0
        sdc['demand_queued'][served_indices] = 0            
    else:
        queued_indices = np.nonzero(sdhl['actual_demand'][type_hc])[0]
        luckyones = random.sample(list(queued_indices), starting_capacity_today)
        unluckyones = queued_indices[~np.isin(queued_indices,luckyones)]
        sdhl['demand_fulfilled_dates'][type_hc][luckyones] = today   
        sdhl['demand_served'][type_hc][today] = sdc['starting_capacity'][today]     
        sdc['demand_unserved'][luckyones] = 0
        sdc['demand_queued'][luckyones] = 0
        sdc['demand_queued'][unluckyones] = 0
        sdhl['bad_experience'][type_hc][unluckyones] = 1
        
    return sdc, sdhl

def total_delay_calc_func(time_window, demand_estimate_range, supply_demand_hc_level):
    index_array = np.array([i for i in range(time_window)])
    
    total_delay = {}
    median_delay = {}
    delay_greater_30 = {}
    frac_vac = {}
    
    for facility in ['phc','ss']:    
#            visit_facility = visit_phc if facility=='phc' else visit_ss
#            total_delay[facility] = [i if j>=0 else np.nan for i,j in zip(supply_demand_hc_level['demand_fulfilled_dates'][facility], visit_facility)]
#            total_delay[facility] = [i-j for i,j in zip(total_delay[facility], index_array)]
        total_delay[facility] = [i-j for i,j in zip(supply_demand_hc_level['demand_fulfilled_dates'][facility], index_array)]
        total_delay[facility] = np.array(total_delay[facility][demand_estimate_range[0]:demand_estimate_range[1]])
        total_delay[facility] = total_delay[facility][total_delay[facility]>=0]
        total_delay[facility] = total_delay[facility][total_delay[facility]!=np.nan]
        median_delay[facility] = np.median(total_delay[facility])
        delay_greater_30[facility] = len(total_delay[facility][total_delay[facility]>30])/len(total_delay[facility])
        
    frac_vac['phc'] = len(total_delay['phc'])/(len(total_delay['phc']) + len(total_delay['ss']))
    frac_vac['ss'] = len(total_delay['ss'])/(len(total_delay['phc']) + len(total_delay['ss']))
    frac_vac['overall'] = (len(total_delay['phc']) + len(total_delay['ss']))/(demand_estimate_range[1]-demand_estimate_range[0])

    median_delay['overall'] = np.median(np.append(total_delay['phc'], total_delay['ss']))
    delay_greater_30['overall'] = (len(total_delay['phc'][total_delay['phc']>30]) + len(total_delay['ss'][total_delay['ss']>30]))/(len(total_delay['phc'])  + len(total_delay['ss']))
    return median_delay, delay_greater_30, frac_vac


# In[2]:
def main(iternum, scen_name, ss_mode, ss_replenishment_source, replenishment_disruption, annual_cce_disruptions):
    
    random.seed(iternum)
        
    time_window = 730
    demand_estimate_range = [150, 150+365]
    
    phc_attributes = {'min_stock_s' : 20, #days of stock
                      'max_stock_S' : 50, #days of stock
                      'phc_replenishment_frequency' : 40, #days
                      'phc_replenishment_disruption' : replenishment_disruption, # Number of disruptions per year
                      'phc_cce_disruption' : random.sample([i for i in range(1, time_window)], annual_cce_disruptions * int(time_window/365))
                     }
    
    ss_attributes = {'max_stock_S':20,
                     'session_frequency': 14,
                     'ss_replenishment_frequency': 28,
                     'ss_replenishment_source': ss_replenishment_source, # phc or dvs
                     'session_start': 21,
                     'ss_replenishment_disruption': replenishment_disruption, # Number of disruptions per year,
                     'ss_cce_disruption' : random.sample([i for i in range(1, time_window)], annual_cce_disruptions * int(time_window/365))
                     }
#    ss_replenishment_frequency = 28
#    session_frequency = 28
    
#    start_date = np.ones(time_window, dtype=np.int)
    
    alternatives = ['NO', 'PHC', 'SS']
    
    
    # ### Attributes of alternatives
    
    # In[3]:
    
    X_distance_phc = np.random.normal(2,1,time_window)
    X_distance_ss = np.random.normal(0,1,time_window)
#    X_days_to_next_session_phc = np.ones(time_window)
    
    if ss_mode == 'periodic':
        # Days to next session at SS, assuming first session is at t=7
        X_days_to_next_session_ss = np.array([ss_attributes['session_frequency'] - (today+7)%ss_attributes['session_frequency']-1 for today in range(time_window)])
    else:
        # if continuous availability at SS, then days to next session = 0
        X_days_to_next_session_ss = np.zeros(time_window)
    
    # ### Attributes of decision makers
    
    # In[4]:
    
    S_sociodemographic = np.random.normal(0,1,time_window)
    
    S_log_days_since_due_date = np.zeros((time_window,time_window))
    
    # S_log_days_since_due_date[n,t]
    # Column represents the day of year, row represents child born on that day
    # R1-C1 = Days since due date on day 1 for child due on day 1
    # R1-C2 = Days since due date on day 2 for child due on day 1
    # R3-C1 = Days since due date on day 1 for child due on day 3 (This is irrelevant, because the child isn't due yet)
    for i in range(time_window):
        S_log_days_since_due_date[i,i:] = [np.log(i+1) for i in range(time_window-i)]
    
    
    # ### Error term
    
    # In[5]:
    
    sigma_sq = 1
    sigma_corr = 0.8
    
    mean = [0] * time_window
    cov = np.ones((time_window, time_window))*sigma_corr + np.diag(np.full(time_window,sigma_sq-sigma_corr)) 
    
    # Sigma[i,n,t]
    sigma_matrix = np.zeros((len(alternatives),time_window,time_window))
    
    for i in range(len(alternatives)):
        sigma_matrix[i,:,:] = np.random.multivariate_normal(mean, cov, time_window)
    
    
    # ### p matrix creation
    
    # In[6]:
    
    beta_1 = -0.6 # ASC PHC
    beta_2 = -0.5 # ASC SS 
    beta_3 = -0.55 # Dist PHC
    beta_4 = -0.4 # Dist SS
    beta_5 = -0.15 # Days to next session at SS
    beta_6 = 0.5 # Sociodemographic
    beta_7 = -0.5 # Prior unserved at PHC (0/1)
    beta_8 = -0.5 # Prior unserved at SS (0/1)
    beta_9 = 0.1 # Days since due date
    
    beta_matrix = np.array(([0,0,0,0,0,0,0,0,0], 
                            [beta_1,0,beta_3,0,0,beta_6,beta_7,0,beta_9], 
                            [0,beta_2,0,beta_4,beta_5,beta_6,0,beta_8,beta_9])).T
    
    uno = np.ones(time_window)
#    sero = np.zeros(time_window)
    
    attribute_matrix = np.zeros((time_window,10,time_window))
    p_matrix = np.zeros((3,time_window, time_window))
    
    
    # In[13]:
    
    supply_demand_combined = {'demand_unserved': np.ones(time_window, dtype=np.int),
                              'demand_queued': np.zeros(time_window, dtype=np.int),
                              'starting_capacity': np.zeros(time_window, dtype=np.int)
                             }
    # supply_demand_combined['starting_capacity'] is used for periodic replenishment at SS
    
    supply_demand_hc_level = {'actual_demand': {'phc':np.array([]), 'ss':np.array([])},
                              'demand_served': {'phc':np.zeros(time_window, dtype=np.int), 'ss':np.zeros(time_window, dtype=np.int)},
                              'bad_experience': {'phc':np.zeros(time_window, dtype=np.int), 'ss':np.zeros(time_window, dtype=np.int)},
                              'demand_fulfilled_dates': {'phc':np.zeros(time_window, dtype=np.int), 'ss':np.zeros(time_window, dtype=np.int)},
                              'starting_capacity': {'phc': np.zeros(time_window, dtype=np.int), 'ss': np.zeros(time_window, dtype=np.int)}
                             }
    # supply_demand_hc_level['starting_capacity'] is used for continuous replenishment at SS
    
    
    # In[14]:
    
#    start_time = time.time()
    #Iterate through each time period
    for i in range(time_window):
    
        attribute_matrix = np.array(([uno, uno, X_distance_phc, X_distance_ss, X_days_to_next_session_ss, S_sociodemographic, supply_demand_hc_level['bad_experience']['phc'], supply_demand_hc_level['bad_experience']['ss'], S_log_days_since_due_date[:,i]])).T
        
        p_matrix = create_p_matrix(i, attribute_matrix, beta_matrix, sigma_matrix, p_matrix)
        
        supply_demand_hc_level = demand_estimation(i, p_matrix, supply_demand_combined, supply_demand_hc_level)

        if ss_mode == 'periodic':        
            supply_demand_combined = starting_capacity_today_func_periodic(i, phc_attributes, supply_demand_combined, supply_demand_hc_level)     
            supply_demand_combined, supply_demand_hc_level = supply_demand_periodic(i, ss_attributes['session_frequency'], supply_demand_combined, supply_demand_hc_level, type_hc = 'ss')
            supply_demand_combined, supply_demand_hc_level = supply_demand_periodic(i, ss_attributes['session_frequency'], supply_demand_combined, supply_demand_hc_level, type_hc= 'phc')
        else:
            supply_demand_hc_level = starting_capacity_today_func_continuous(i, phc_attributes, ss_attributes, supply_demand_hc_level)
            supply_demand_combined, supply_demand_hc_level = supply_demand_continuous(i, ss_attributes['session_frequency'], supply_demand_combined, supply_demand_hc_level, type_hc= 'ss')
            supply_demand_combined, supply_demand_hc_level = supply_demand_continuous(i, ss_attributes['session_frequency'], supply_demand_combined, supply_demand_hc_level, type_hc= 'phc')
    
#    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    # In[15]:
    
    plt.style.use("dark_background")
    plt.figure(figsize=(12, 4))
    plt.plot([i for i in range(time_window)],supply_demand_combined['starting_capacity'])
    plt.ylabel('Starting Capacity')
    plt.xlabel('Day')
    plt.show()
    
    
    # In[16]:
    
    visit_phc = np.zeros(time_window, dtype=np.int)
    visit_ss = np.zeros(time_window, dtype=np.int)
    visit_phc_or_ss = np.zeros(time_window, dtype=np.int)
    
    for i in range(time_window):
        # Find all dates AFTER DUE DATE when PHC/SS was chosed by individual i
        # If no such dates then -1, else return first date 
        # This represents number of days since due date when HC was selected
        visit_phc[i] = -1 if len(np.where(np.argmax(p_matrix[:,i,i:], axis = 0)==1)[0])==0 else int(np.where(np.argmax(p_matrix[:,i,i:], axis = 0)==1)[0][0])
        visit_ss[i] = -1 if len(np.where(np.argmax(p_matrix[:,i,i:], axis = 0)==2)[0])==0 else int(np.where(np.argmax(p_matrix[:,i,i:], axis = 0)==2)[0][0])
        visit_phc_or_ss[i] = -1 if len(np.where(np.argmax(p_matrix[:,i,i:], axis = 0)!=0)[0])==0 else int(np.where(np.argmax(p_matrix[:,i,i:], axis = 0)!=0)[0][0])
        
        # If PHC chosen first then visit_SS = -1 else otherwise
        if (visit_phc[i]!=-1) & (visit_ss[i]!=-1):
            if visit_phc[i]>visit_ss[i]:
                visit_phc[i] = -1
            elif visit_phc[i]<visit_ss[i]:
                visit_ss[i] = -1    
    
    
    # In[17]:
    
#    print(f'Number of PHC choice : {np.sum((visit_phc!=-1)*1)}')
#    print(f'Number of SS choice : {np.sum((visit_ss!=-1)*1)}')
#    print(f'\nFraction of PHC choice : {np.sum((visit_phc!=-1)*1)/time_window}')
#    print(f'Fraction of SS choice : {np.sum((visit_ss!=-1)*1)/time_window}')
#    print(f'Fraction not vaccinating : {1 - np.sum((visit_phc_or_ss!=-1)*1)/time_window}')
##    
#    
#    # In[18]:
#    
#    print(f"Number served at PHC : {np.sum(supply_demand_hc_level['demand_served']['phc'])}")
#    print(f"Number served at SS : {np.sum(supply_demand_hc_level['demand_served']['ss'])}")
    
    
    # In[19]:
    
#    choice_ss = list((visit_ss!=-1).nonzero()[0])
#    served_ss = list(np.nonzero(supply_demand_hc_level['demand_fulfilled_dates']['ss'])[0])
#    print(f'Choice SS but not served at SS: {list(set(choice_ss) - set(served_ss))}')
#    print(f'Choice not SS but served at SS: {list(set(served_ss) - set(choice_ss))}')
    
    
    # Obtain Choice delay, Supply delay and Total delay
    
    # In[69]:
        
    
    
    median_delay , delay_greater_30, frac_vac = total_delay_calc_func(time_window, demand_estimate_range, supply_demand_hc_level)
    return median_delay, delay_greater_30, frac_vac
#    choice_delay = [i if i>=0 else j for i, j in zip(visit_ss,visit_phc)]
#    
#    total_delay = [i if i>0 else j for i, j in zip(supply_demand_hc_level['demand_fulfilled_dates']['ss'],supply_demand_hc_level['demand_fulfilled_dates']['phc'])]
#    
#    index_array = np.array([i for i in range(time_window)])
#    
#    total_delay_phc = [i-j if i!=0 else 0 for (i,j) in zip(supply_demand_hc_level['demand_fulfilled_dates']['phc'], index_array)]
#    total_delay_ss = [i-j if i!=0 else 0 for (i,j) in zip(supply_demand_hc_level['demand_fulfilled_dates']['ss'], index_array)]
#    
#    supply_delay_phc = np.array([i-j if ((k!=0)&(j!=-1)) else 0 for (i,j,k) in zip(total_delay_phc, visit_phc, supply_demand_hc_level['demand_fulfilled_dates']['phc'])])
#    supply_delay_ss = np.array([i-j if ((k!=0)&(j!=-1)) else 0 for (i,j,k) in zip(total_delay_ss, visit_ss, supply_demand_hc_level['demand_fulfilled_dates']['ss'])])
#    
#    
#    # In[70]:
#    
#    out_choice_delay_phc = visit_phc[demand_estimate_range[0]:demand_estimate_range[1]]
#    out_choice_delay_ss = visit_ss[demand_estimate_range[0]:demand_estimate_range[1]]
#    
#    out_choice_delay_phc = out_choice_delay_phc[np.where(out_choice_delay_phc>=0)]
#    out_choice_delay_ss = out_choice_delay_ss[np.where(out_choice_delay_ss>=0)]
#    
#    out_supply_delay_phc = supply_delay_phc[demand_estimate_range[0]:demand_estimate_range[1]]
#    out_supply_delay_ss = supply_delay_ss[demand_estimate_range[0]:demand_estimate_range[1]]
#    
#    out_supply_delay_phc = out_supply_delay_phc[np.where(out_supply_delay_phc>=0)]
#    out_supply_delay_ss = out_supply_delay_ss[np.where(out_supply_delay_ss>=0)]
#
#    append_num = phc_replenishment_disrupt if scen_name == "replenishmentdisruption" else annual_cce_disruptions
#    
#    out_hist_choice_delay_phc = np.reshape(np.append(np.histogram(out_choice_delay_phc, bins=np.arange(0,365,30))[0], append_num), (1, len(np.histogram(out_choice_delay_phc, bins=np.arange(0,365,30))[0])+1))
#    out_hist_choice_delay_ss = np.reshape(np.append(np.histogram(out_choice_delay_ss, bins=np.arange(0,365,30))[0], append_num), (1, len(np.histogram(out_choice_delay_ss, bins=np.arange(0,365,30))[0])+1))
#    out_hist_supply_delay_phc = np.reshape(np.append(np.histogram(out_supply_delay_phc, bins=np.arange(0,365,30))[0], append_num), (1, len(np.histogram(out_supply_delay_phc, bins=np.arange(0,365,30))[0])+1))
#    out_hist_supply_delay_ss = np.reshape(np.append(np.histogram(out_supply_delay_ss, bins=np.arange(0,365,30))[0], append_num), (1, len(np.histogram(out_supply_delay_ss, bins=np.arange(0,365,30))[0])+1))
#    
#      
#    # In[71]:
#    
#    with open(path_data / f'07 out_choice_delay_phc_{scen_name}_{ss_mode}.csv', 'a') as f:
#        np.savetxt(f, out_hist_choice_delay_phc, fmt='%s', delimiter=',', newline='\n')
#    with open(path_data / f'07 out_choice_delay_ss_{scen_name}_{ss_mode}.csv', 'a') as f:
#        np.savetxt(f, out_hist_choice_delay_ss, fmt='%s', delimiter=',', newline='\n')    
#    with open(path_data / f'07 out_supply_delay_phc_{scen_name}_{ss_mode}.csv', 'a') as f:
#        np.savetxt(f, out_hist_supply_delay_phc, fmt='%s', delimiter=',', newline='\n')    
#    with open(path_data / f'07 out_supply_delay_ss_{scen_name}_{ss_mode}.csv', 'a') as f:
#        np.savetxt(f, out_hist_supply_delay_ss, fmt='%s', delimiter=',', newline='\n')


# In[38]:


if __name__ == "__main__":
    main()


#df2 = pd.DataFrame({'starting_capacity': supply_demand_combined['starting_capacity'],
#                    'choice_delay_phc': visit_phc,
#                    'choice_delay_ss': visit_ss,
#                    'demand_served_phc': supply_demand_hc_level['demand_served']['phc'],
#                    'demand_served_ss': supply_demand_hc_level['demand_served']['ss'],
#                    'demand_fulfilled_dates_phc' : supply_demand_hc_level['demand_fulfilled_dates']['phc'],
#                    'demand_fulfilled_dates_ss' : supply_demand_hc_level['demand_fulfilled_dates']['ss'],
#                    'bad_experience_phc': supply_demand_hc_level['bad_experience']['phc'],
#                    'bad_experience_ss': supply_demand_hc_level['bad_experience']['ss']})
#
#
## In[39]:
#
#df2.to_csv(path_data / '07 choice delay vs total_delay - all metrics - 30082024.csv')
#
#
## In[ ]:
#
#df = pd.DataFrame( {'NO': p_matrix[0,59,:], 'PHC': p_matrix[1,59,:], 'SS': p_matrix[2,59,:]})
#df.to_csv(path_data / 'p_matrix_59.csv')