# coding: utf-8

import numpy as np
from scipy import stats
import random
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('grayscale')

len_time_window = 1000
min_window = 100
max_window = 900

session_frequency = 15
    
daily_birth_lambda = 4/3

s_values = np.arange(1,41)
#s_values = [15]


max_backorders = 40

n_iter = 100

hist_range = 150

metrics_df = pd.DataFrame({'metric':[],'s':[],'bottom5':[], 'mean':[], 'top5':[]})
histogram_df = pd.DataFrame({'s':[],'bin':[], 'bottom5':[], 'mean':[], 'top5':[]})
cdf_df = pd.DataFrame({'s':[],'bin':[], 'bottom5':[], 'mean':[], 'top5':[]})

session_days = [i+5 for i in range(len_time_window) if i%session_frequency==0]
session_days_cropped =  [i+5 for i in range(min_window, max_window) if i%session_frequency==0]

for s in s_values:
    A_lst = []
    B_lst = []
    p_lst = []
    coverage_lst = []
    hist_array = np.zeros((n_iter,hist_range))
    sessions_delayed_array = np.zeros((n_iter,max_backorders+1))

    for iter_num in range(n_iter):
        daily_births_array = np.random.poisson(daily_birth_lambda, size=len_time_window)
        
        ctr=0
        child_bday_dict={}
        for idx,i in enumerate(daily_births_array):
            for j in range(i):
                child_bday_dict.update({ctr:idx})
                ctr+=1
        
        inaction_end_day = [[idx+stats.expon.rvs(loc=0, scale=10) for j in range(i)] for idx, i in enumerate(daily_births_array)]
        
        flat_inaction_end_day = [x for xs in inaction_end_day for x in xs]
                
        new_child_session_day = [min([sday for sday in session_days if sday>=iday], default="EMPTY") for xs in inaction_end_day for iday in xs]
        
        actual_session_day = np.array([i for i in new_child_session_day if i!="EMPTY"])
        
        num_sessions_delayed = np.zeros(len(new_child_session_day))
        
        covered = 0
        uncovered = 0
        
        A = [0]*len(session_days_cropped)
        B = [0]*len(session_days_cropped)
        B_copy = [0]*len(session_days)
        p = [0]*len(session_days_cropped)
        pi = [0]*(max_backorders+1)
        
        s_idx_cropped = 0
        
        for s_idx,s_day in enumerate(session_days):
            s_children = [idx for idx,i in enumerate(actual_session_day) if i==s_day]
            vaxd_children = random.sample(s_children, int(min(s+B_copy[s_idx-1], len(s_children))))
            unvaxd_children = list(set(s_children)-set(vaxd_children))
            num_sessions_delayed[unvaxd_children]+=1
            over_backordered_children = np.array([idx for idx, i in enumerate(num_sessions_delayed) if i>max_backorders])
            actual_session_day[unvaxd_children] = actual_session_day[unvaxd_children]+session_frequency
            B_copy[s_idx] = len(unvaxd_children)
            if len(over_backordered_children)>0:
                num_sessions_delayed[over_backordered_children]=-1
                actual_session_day[over_backordered_children] = -1
            if (s_day>min_window) and (s_day<max_window):
                A[s_idx_cropped] = 1 if len(unvaxd_children)>0 else 0
                B[s_idx_cropped] = len(unvaxd_children)
                p[s_idx_cropped] = len(vaxd_children)/(len(vaxd_children)+len(unvaxd_children))
                covered += len(vaxd_children)
                uncovered += len(over_backordered_children)
                s_idx_cropped +=1
            del over_backordered_children

        total_delay = [actual_session_day[idx] - child_bday_dict[idx] + random.uniform(0,1) for idx,i in enumerate(actual_session_day) if (i>100)and(i<900)]  
        hist_array[iter_num] = np.histogram(total_delay, range(hist_range+1),weights=np.ones(len(total_delay)) / len(total_delay))[0]
#       
#        try:
#            sessions_delayed_array[iter_num] = np.unique(num_sessions_delayed, return_counts = True)[1][1:]/sum(np.unique(num_sessions_delayed, return_counts = True)[1][1:])
#        except ValueError:
#            sessions_delayed_array[iter_num] = np.unique(num_sessions_delayed, return_counts = True)[1]/sum(np.unique(num_sessions_delayed, return_counts = True)[1])
        
        if np.max(num_sessions_delayed)==max_backorders:
            print("Max num sessions exceeded: ",iter_num)
           
        A_lst.append(np.mean(A))
        B_lst.append(np.mean(B))
        p_lst.append(np.mean(p))
        coverage_lst.append(covered/(covered+uncovered))
#    
    metrics_df = metrics_df.append({'metric': 'A','s':s,'bottom5': np.percentile(A_lst,5), 'mean':np.percentile(A_lst,50), 'top5':np.percentile(A_lst,95)},ignore_index=True)
    metrics_df = metrics_df.append({'metric': 'B','s':s,'bottom5': np.percentile(B_lst,5), 'mean':np.percentile(B_lst,50), 'top5':np.percentile(B_lst,95)},ignore_index=True)
    metrics_df = metrics_df.append({'metric': 'p','s':s,'bottom5': np.percentile(p_lst,5), 'mean':np.percentile(p_lst,50), 'top5':np.percentile(p_lst,95)},ignore_index=True)
    metrics_df = metrics_df.append({'metric': 'Coverage','s':s,'bottom5': np.percentile(coverage_lst,5), 'mean':np.percentile(coverage_lst,50), 'top5':np.percentile(coverage_lst,95)},ignore_index=True)
    
#    histogram_df = histogram_df.append(pd.DataFrame({'s':[s]*hist_range, 'bin':np.arange(hist_range), 'bottom5':np.percentile(hist_array, 5, axis=0), 'mean':np.mean(hist_array, axis=0), 'top5':np.percentile(hist_array, 95, axis=0)}),ignore_index=True)
    cdf_array = np.cumsum(hist_array, axis = 1)
    cdf_df = cdf_df.append(pd.DataFrame({'s':[s]*hist_range, 'bin':np.arange(hist_range), 'bottom5':np.percentile(cdf_array, 5, axis=0), 'mean':np.percentile(cdf_array, 50, axis=0), 'top5':np.percentile(cdf_array, 95, axis=0)}),ignore_index=True)
    cdf_df.drop('s', axis = 1).to_csv(r"C:\Users\Rohan Bapat\Documents\Projects\Immunization Supply Chain\immunization_supply_chain\inputs\total_delay_cdf_perfect_backlogging_s15_kappa10.csv", index = False, header=False)
#    np.histogram(total_delay, bins = np.arange(0,45,5))
#    
#    
#    
#    n_delay_idx = [idx for idx,i in enumerate(num_sessions_delayed) if i==n_delay][:90000]
#    
#    
#    plt.hist(total_delay, range(min(total_delay), max(total_delay) + 5, 5))
#    plt.show()
#    
#    
#    np.max(num_sessions_delayed)
        
#plt.figure(figsize=(12,8))
#plt.plot(cdf_df['bin'], cdf_df['mean'])
#plt.plot(cdf_df['bin'], cdf_df['bottom5'],':')
#plt.plot(cdf_df['bin'], cdf_df['top5'],':')
#plt.show()
#
#plt.plot(np.arange(1,31),metrics_df.loc[metrics_df['metric']=='Coverage','mean'],color = 'black')
#plt.show()
#
#plt.bar(np.arange(0,200,5),np.mean(hist_array, axis=0),color = 'black')
#plt.show()
    
    
pi_low5 = np.percentile(sessions_delayed_array, 5, axis =0)
pi_mean = np.mean(sessions_delayed_array, axis = 0)
pi_high5 = np.percentile(sessions_delayed_array, 95, axis =0)

plt.plot(np.arange(max_backorders+1), pi_low5, color = 'red', linestyle= '-.') 
plt.plot(np.arange(max_backorders+1), pi_mean, color = 'red', linestyle= '-') 
plt.plot(np.arange(max_backorders+1), pi_high5, color = 'red', linestyle= '-.') 
plt.plot(np.arange(max_backorders+1),[0.955256, 0.0428242, 0.00191981], color = 'black')
plt.xticks(np.arange(max_backorders+1))
plt.xlabel('Number of sessions delayed')
plt.ylabel('Pi')
plt.show()

lamT = 20
A_bar = []
B_bar = []
p_bar = []
p_bar_2 = []

for s in s_values:
    A_bar.append(1-stats.poisson.cdf(k=s, mu=lamT))
    b_mean = lamT - sum([1-stats.poisson.cdf(k=j, mu = lamT) for j in range(s)])
    B_bar.append(b_mean)
    p_bar.append(min((s+b_mean)/(lamT+b_mean),1))
    p_bar_2.append(1-b_mean/(lamT+b_mean))
    

plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='A','mean'], color = 'red', linestyle= '-') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='A','top5'], color = 'red', linestyle= '-.') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='A','bottom5'], color = 'red',linestyle= '-.') 
plt.plot(s_values,A_bar,color = 'black')
plt.xlabel('s')
plt.ylabel('A')
plt.grid()
plt.show()

plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='B','mean'], color = 'red', linestyle= '-') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='B','top5'], color = 'red', linestyle= '-.') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='B','bottom5'], color = 'red',linestyle= '-.') 
plt.plot(s_values,B_bar,color = 'black')
plt.xlabel('s')
plt.ylabel('B')
plt.grid()
plt.show()
    
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='p','mean'], color = 'red', linestyle= '-') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='p','top5'], color = 'red', linestyle= '-.') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='p','bottom5'], color = 'red',linestyle= '-.') 
plt.plot(s_values,p_bar_2,color = 'black')
plt.xlabel('s')
plt.ylabel('p')
plt.grid()
plt.show()

plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='Coverage','mean'], color = 'red', linestyle= '-') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='Coverage','top5'], color = 'red', linestyle= '-.') 
plt.plot(s_values,metrics_df.loc[metrics_df['metric']=='Coverage','bottom5'], color = 'red',linestyle= '-.') 
plt.xlabel('s')
plt.ylabel('Coverage')
plt.grid()
plt.show()