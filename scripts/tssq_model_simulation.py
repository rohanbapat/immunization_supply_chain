
# coding: utf-8

# In[1]:

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('grayscale')
import random

import pandas as pd


# In[2]:


def main(sim_iter):
    
    # Parameters
    random.seed(sim_iter)    
    len_time_window = 1000    
    n_sessionsites = 2    
    annual_births = [300, 300]
    tau = 10
    eta = 15
    session_frequency = 15
    session_start = [5, 10]
    phc_replenishment_frequency = 30
    service_level = 0.9
    emergency_replenishment_volume = "S" # "S" or "Expected Demand"
       
    #Session details
    session_days = [None]*n_sessionsites
    session_days[0] = [i for i in range(len_time_window) if (i-session_start[0])%session_frequency==0]
    session_days[1] = [i for i in range(len_time_window) if (i-session_start[1])%session_frequency==0]
    periodic_replenishment_days = [i for i in range(len_time_window) if i%phc_replenishment_frequency==0] 
    
    e_session_demand = [annual_births[i]/365*session_frequency for i in range(n_sessionsites)]
    session_carryvolume = [stats.poisson.ppf(service_level,annual_births[i]/365*session_frequency,0) for i in range(n_sessionsites)]
    S = sum(annual_births)/12*1.5 
    s = sum(annual_births)/12*0.5 

    # ### Demand
    
    daily_births_array = [None]*n_sessionsites
    child_bday_dict = [{} for _ in range(n_sessionsites)]  ### Initialized differently because dict were getting copied
    inaction_end_day = [None]*n_sessionsites
    flat_inaction_end_day = [None]*n_sessionsites
    new_child_session_day = [None]*n_sessionsites
    
    for site in range(n_sessionsites):
        daily_births_array[site] = np.random.poisson(annual_births[site]/365, size=len_time_window)
        ctr=0
        for idx,i in enumerate(daily_births_array[site]):
            for j in range(i):
                child_bday_dict[site].update({ctr:idx})
                ctr+=1
        
        # This array gives day of end of inaction period for each child by site
        inaction_end_day[site] = [[np.floor(idx+stats.expon.rvs(loc=0, scale=tau)) for j in range(i)] for idx, i in enumerate(daily_births_array[site])]
    
        # Flatten above array
        flat_inaction_end_day[site] = [x for xs in inaction_end_day[site] for x in xs]
    
        # Based on above array, identify the subsequent session day 
        new_child_session_day[site] = [min([sday for sday in session_days[site] if sday>iday], default="EMPTY") for xs in inaction_end_day[site] for iday in xs]
    
#    one_month_of_supply = sum(annual_births)/12
    
    # Initialize variables for simulation
    
    starting_capacity = np.zeros(len_time_window)
    session_new_demand = np.zeros(len_time_window)
    session_total_demand = np.zeros(len_time_window)
    demand_served = np.zeros(len_time_window)
    demand_unserved = np.zeros((n_sessionsites, len_time_window))
    unserved_child_dict = [{} for _ in range(n_sessionsites)]  ### Initialized differently because dict were getting copied
    retrial_child_session_day = [[None]*len(new_child_session_day[i]) for i in range(n_sessionsites)] 
    ending_capacity = np.zeros(len_time_window)
    sessions_till_next_replenishment = [None]*n_sessionsites
    session_days = [None]*n_sessionsites
    child_vacday_dict = [{} for _ in range(n_sessionsites)]
    
    session_days[0] = [i for i in range(len_time_window) if (i-session_start[0])%session_frequency==0]
    session_days[1] = [i for i in range(len_time_window) if (i-session_start[1])%session_frequency==0]
    periodic_replenishment_days = [i for i in range(len_time_window) if i%phc_replenishment_frequency==0]
                          
    for day in range(len_time_window-1):
    
        # Periodic replenishment
        if day%phc_replenishment_frequency==0:
            starting_capacity[day] = S
        
        # Emergency replenishment
        if starting_capacity[day] < s:
            # Check if emergency replenishment is required
            next_replenishment = min([l for l in periodic_replenishment_days if l>=day])
            sessions_till_next_replenishment[0] = [k for k in session_days[0] if ((k>=day) and (k<next_replenishment))]
            sessions_till_next_replenishment[1] = [k for k in session_days[1] if ((k>=day) and (k<next_replenishment))]
    
            if len(sessions_till_next_replenishment[0]+sessions_till_next_replenishment[1])>0:
                if emergency_replenishment_volume=="S":
                    starting_capacity[day] = S
                else:
                    expected_demand = e_session_demand[0]*len(sessions_till_next_replenishment[0])+e_session_demand[1]*len(sessions_till_next_replenishment[1])
                    starting_capacity[day] += expected_demand
        
        for ss in range(n_sessionsites):
            if (day-session_start[ss])%session_frequency==0:
                # If session day at 'ss' session site
                new_child_idx_attending_today = [idx for idx,i in enumerate(new_child_session_day[ss]) if i==day]            
                re_child_idx_attending_today = [idx for idx,i in enumerate(retrial_child_session_day[ss]) if i==day]
            
                all_child_idx_attending_today = new_child_idx_attending_today + re_child_idx_attending_today
                session_new_demand[day] = len(new_child_idx_attending_today)
                session_total_demand[day] = len(all_child_idx_attending_today)
                demand_served[day] = min(starting_capacity[day], session_total_demand[day], session_carryvolume[ss])
                demand_unserved[ss,day] = max(session_total_demand[day] - demand_served[day], 0)
                
                unserved_child_today = []  # Initialize in case no unserved today
                if demand_unserved[ss,day]>0:
                    unserved_child_today = random.sample(all_child_idx_attending_today, int(demand_unserved[ss,day]))
                    unserved_child_dict[ss].update({day:unserved_child_today})
                    retrial_inaction_end_day = list(day+stats.expon.rvs(loc=0, scale=eta,size = int(demand_unserved[ss,day])))
                    
                    for uchild_idx, uchild in enumerate(unserved_child_today):
                        retrial_child_session_day[ss][uchild] = min([sday for sday in session_days[ss] if sday>retrial_inaction_end_day[uchild_idx]], default="EMPTY")
                
    
                served_child_today = list(set(all_child_idx_attending_today)-set(unserved_child_today))
                        
                child_vacday_dict[ss].update({i:day for i in served_child_today})
                ending_capacity[day] = starting_capacity[day] - demand_served[day]
                starting_capacity[day+1] = ending_capacity[day]
            else:
                demand_unserved[ss,day+1] = demand_unserved[ss,day]
        
        if demand_served[day]==0:
            ending_capacity[day] = starting_capacity[day]
            starting_capacity[day+1] = starting_capacity[day]
    
    
    # Outputs
    
    vaccination_delay_calc_list = [None]*n_sessionsites
    vaccination_delay_histogram = [None]*n_sessionsites
    frac_sessions_partially_fulfilled = [None]*n_sessionsites
    frac_demand_retrialled = [None]*n_sessionsites
    
    
    for ss in range(n_sessionsites):
        vaccination_delay_calc_list[ss] = [child_vacday_dict[ss][k] - child_bday_dict[ss][k] for k,v in child_vacday_dict[ss].items()]
        vaccination_delay_histogram[ss] = np.histogram(vaccination_delay_calc_list[ss], bins = list(range(0,101,5)), weights=np.ones(len(vaccination_delay_calc_list[ss])) / len(vaccination_delay_calc_list[ss]))
        frac_sessions_partially_fulfilled[ss] = len(unserved_child_dict[ss].keys())/len(session_days[ss])
        frac_demand_retrialled[ss] = len([x for xs in list(unserved_child_dict[ss].values()) for x in xs])/len(child_vacday_dict[ss].keys())
    
    return vaccination_delay_histogram, frac_sessions_partially_fulfilled, frac_demand_retrialled
    
if __name__ == "__main__":
    main()
   
    
