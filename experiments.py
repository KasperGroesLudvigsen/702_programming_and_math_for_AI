# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:52:59 2020

@author: groes
"""
import task1_class_method_def_NEW_APPROACH_backup0712 as t1
import numpy as np
import pandas as pd
import random
import timeit
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing

##############################################################################
################### FOR RUNNING MULTIPLE EXPERIMENTS EASILY ##################
##############################################################################

def experiments(grid_height, grid_width, ants, rho, alpha, beta):
    
    
    ########################### INITIALIZING THE GRID ############################
    
    game = t1.Game(grid_height, grid_width)
    print(game.grid)
    
    ################################ RANDOM ALGO #################################
    """
    print("Running random algo")
    # Im making a random algorithm to show that my baseline algorithm is better 
    # than random
    time_start = time.time()
    
    random_algo = t1.RandomAlgo(game.grid)
    
    path_random_algo, time_spent_random_algo = random_algo.run_random_algo()
    time_end = time.time()
    compute_time_random = time_end-time_start
    steps_in_path_random = len(path_random_algo)
    """
    
    ################################ BASELINE ALGO ###########################
    print("Running baseline algo")

    time_start = time.time()
    baseline_algo = t1.BaselineAlgo(game.grid)
    path_baseline, time_spent_baseline = baseline_algo.run_baseline()
    time_end = time.time()
    compute_time_baseline = time_end-time_start
    steps_in_path_baseline = len(path_baseline)
    
    ################################ DIJKSTRA'S ##############################
    print("Running Dijkstra's algo")

    time_start = time.time()
    dijkstra = t1.Dijkstras(game.grid)
    path_dijkstras, time_spent_dijkstras = dijkstra.run_dijkstras()
    time_end = time.time()
    compute_time_dijkstras = time_end-time_start
    steps_in_path_dijkstras = len(path_dijkstras)
    
    #################################### ACO #################################
    print("Running ACO algo")
    time_start = time.time()
    aco = t1.ACO(game.grid)
    path_aco, time_spent_aco = aco.run_ACO(ants)
    time_end = time.time()
    compute_time_aco = time_end-time_start
    steps_in_path_aco = len(path_aco)
    
    ############################# COLLECTING RESULTS ##########################
    results = {
        "Grid_size" : str(grid_height) + "x" + str(grid_width),
        
        "Algorithm" : [ "Baseline", "Dijkstras", "ACO"],
        
        "Path" : [path_baseline, path_dijkstras, path_aco],
        
        "Nodes_in_path" : [steps_in_path_baseline,
                           steps_in_path_dijkstras, steps_in_path_aco],
        
        "Length_best_path" : [time_spent_baseline,
                              time_spent_dijkstras, time_spent_aco],
        
        "Time_to_compute" : [compute_time_baseline,
                             compute_time_dijkstras, compute_time_aco]
        }
    
    results_df = pd.DataFrame.from_dict(data=results)
    
    return results_df


random.seed(101)
grid_height = 2
grid_width = 3

# To be used in ACO
ants = 100 # number of ants to be sent through the grid
rho = 0.5
alpha = 2
beta = 2

#results_df = main(grid_height, grid_width, ants, rho, alpha, beta)
##all_results = results_df
#all_results = all_results.append(results_df)  #pd.concat(all_results, results_df)
#all_results.to_excel("all_results.xlsx")
  

## Calling main function for each iteration over the grid size lists, aggregating
## all results into one df
gridheights = [2, 3, 4, 5] #[4, 8, 16, 32, 64, 128]
gridwidths = [3, 4, 6, 6] #[5, 10, 20, 40, 80, 160]

for height, width in zip(gridheights, gridwidths):
    results_df = experiments(height, width, ants, rho, alpha, beta)
    all_results = all_results.append(results_df)
    print("next")
#all_results.to_excel("all_results_new.xlsx")


# Adding grid_size column for visualization
#grird_size = [h*w for h, w in zip(gridheights, gridwidths)]
#grid_size_insertion = []
#for i in grird_size:
#    grid_size_insertion.append(i)
#    grid_size_insertion.append(i)
#    grid_size_insertion.append(i)
#all_results["grid_size"] = grid_size_insertion

## PLOTTING
# Seperating algos in separate dfs
aco_results = all_results[all_results.Algorithm.eq("ACO")]
dijkstras_results = all_results[all_results.Algorithm.eq("Dijkstras")]
baseline_results = all_results[all_results.Algorithm.eq("Baseline")]

# Plotting compute time against grid size
ax = sns.lineplot(x="grid_size", y="Time_to_compute", hue="Algorithm", data=aco_results)
ax2 = sns.lineplot(x="grid_size", y="Time_to_compute", hue="Algorithm", data=dijkstras_results)
ax3 = sns.lineplot(x="grid_size", y="Time_to_compute", hue="Algorithm", data=baseline_results)

# Plotting length of path against grid_size
length_vs_gridsize = sns.lineplot(x="grid_size", y="Length_best_path", hue="Algorithm", data=all_results)

# Plotting scaled compute time against grid size 
compute_time_scaled = np.array(all_results["Time_to_compute"])
compute_time_scaled = preprocessing.scale(compute_time_scaled)
all_results["compute_time_scaled"] = compute_time_scaled
ax_compute_time_gridsize = sns.lineplot(x="grid_size", y="compute_time_scaled", hue="Algorithm", data=all_results)

