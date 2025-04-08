# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:20:19 2024

@author: Rohan Bapat
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Example: Load the dataset (replace with actual dataset)
# The dataset should contain columns 'Choice 1', 'Choice 2', 'Choice 3' and exogenous variables 'x1', 'x2', ..., 'xk'
data = pd.read_csv(r'C:\Users\Rohan Bapat\Documents\Projects\Immunization Supply Chain\immunization_supply_chain\data\09 choices for MLE sample v2.csv')  # Replace with actual filename

# Extract the relevant parts of the data
n = len(data)  # Number of decision-makers
choice_columns = ['ChosenNO', 'ChosenPHC', 'ChosenSS']
exogenous_columns = ['uno1', 'uno2', 'dist_phc', 'dist_ss', 'days_to_next_session_ss', 'sociodemographic', 'prior_unserved_phc', 'prior_unserved_ss', 'log_days_since_due_date']  # All other columns are exogenous variables
num_params = len(exogenous_columns)  # Number of exogenous variables

# Chosen alternative indicators (n x 3)
choices = data[choice_columns].values  # This will be a matrix with binary 1/0 indicating chosen alternative

# Exogenous variables matrix (n x k)
X = data[exogenous_columns].values

# Add a column of ones for the intercept
#X = np.hstack([np.ones((n, 1)), X])  # n x (k + 1)

# Log-likelihood function
def log_likelihood(beta, choices, X):
    """
    Log-likelihood function for the multinomial logit model.

    params: array of parameters to estimate
    choices: binary matrix of choices (n x 3)
    X: exogenous variables (n x k+1)

    Returns the negative log-likelihood.
    """
    # Extract alternative-specific constants (ASCs) and betas
    beta_matrix = np.array(([0,0,0,0,0,0,0,0,0], 
                        [beta[0],0,beta[2],0,0,beta[5],beta[6],0,beta[8]], 
                        [0,beta[1],0,beta[3],beta[4],beta[5],0,beta[7],beta[8]])).T

    v_matrix = np.matmul(X, beta_matrix)
    
    exp_v_matrix = np.exp(v_matrix)
    
    choice_probabilities = exp_v_matrix / np.sum(exp_v_matrix, axis=1, keepdims=True)

    # Log-likelihood calculation
    log_likelihood_value = np.sum(choices * np.log(choice_probabilities))

    # Return negative log-likelihood for minimization
    return -log_likelihood_value

# Initial parameter guesses: 0 for all parameters (2 ASCs + k betas)
initial_betas = np.zeros(num_params)
#initial_betas = [-0.6, -0.5, -0.55, -0.4, -0.15, 0.5, -0.5, -0.5, 0.1]

# Minimize the negative log-likelihood using BFGS
#result = minimize(log_likelihood, initial_betas, args=(choices, X), method='BFGS')
result = minimize(log_likelihood, initial_betas, args=(choices, X), method = 'Nelder-Mead')

# Print the optimization result
print("Optimization Success:", result.success)
print("Optimization Message:", result.message)
print("Number of iterations:", result.nit)
print("Estimated Parameters:", np.array(result.x))
print("Log-likelihood at convergence:", -result.fun)

# The estimated parameters are in result.x
