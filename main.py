# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:44:28 2020

@author: groes
"""

import import_ipynb # if import fails, open command prompt and run the command "pip install import_ipynb"
import class_definitions as t1
import random
import time


if __name__ == "__main__":
    # Initialize game object to generate grid
    random.seed(101)
    game = t1.Game(12, 10)
    print("Grid size: {}".format(game.grid.shape))
    print(game.grid)
    
    ####################### RUN ACO SEPARATELY ##########################
    aco = t1.ACO(game.grid)
    ants = 100
    rho = 0.5
    alpha = 2
    beta = 2
    print("Running ACO algorithm with {} ants".format(ants))
    
    start = time.time()
    shortest_path, length_shortest_path = aco.run_ACO(ants, rho, alpha, beta)
    end = time.time()
    compute_time = end-start
    
    print("Length of ACO path: {}".format(length_shortest_path))
    print("Compute time of ACO: {}".format(compute_time))
       
    ##################### RUN DIJKSTRAS SEPARATELY ###############################
    print("Running Dijkstra's algorithm")
    dijkstra = t1.Dijkstras(game.grid)
    time_start = time.time()
    path_dijkstras, time_spent_dijkstras = dijkstra.run_dijkstras()
    time_end = time.time()
    compute_time_dijkstras = time_end-time_start
    steps_in_path_dijkstras = len(path_dijkstras)
    print("Length of Dijkstra's path: {}".format(time_spent_dijkstras))
    print("Compute time of Dijkstra's: {}".format(compute_time_dijkstras))
    
    ##################### RUN BASELINE SEPARATELY ###############################
    print("Running baseline algorithm")
    baseline_algo = t1.BaselineAlgo(game.grid)
    time_start = time.time()
    path_baseline, time_spent_baseline = baseline_algo.run_baseline()
    time_end = time.time()
    compute_time_baseline = time_end-time_start
    steps_in_path_baseline = len(path_baseline)
    print("Length of baseline path: {}".format(time_spent_baseline))
    print("Compute time of baseline: {}".format(compute_time_baseline))

    ##################### RUN RANDOM SEPARATELY ###############################
    print("Running random algorithm")
    # Im making a random algorithm to show that my baseline algorithm is better 
    # than random
    random_algo = t1.RandomAlgo(game.grid)
    time_start = time.time()
    path_random_algo, time_spent_random_algo = random_algo.run_random_algo()
    time_end = time.time()
    compute_time_random = time_end-time_start
    steps_in_path_random = len(path_random_algo)
    print("Length of random path: {}".format(time_spent_random_algo))
    print("Compute time of random algorithm: {}".format(compute_time_random))
