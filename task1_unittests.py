# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:29:54 2020

@author: groes
"""
import class_definitions as t1
import numpy as np
import pandas as pd
from unittest.mock import patch
import math
from statistics import mode

height = 10
width = 12
game = t1.Game(height, width)
test_grid = game.create_grid()

def unittest_create_grid():
    assert test_grid.shape[0] == height
    assert test_grid.shape[1] == width
    for row in test_grid:
        assert np.min(row) >= 0
        assert np.max(row) <= 9

unittest_create_grid()


def unittest_baseline_algo():
    test_grid = np.array([[1,1,1], [1,1,1]])
    time_spent = game.baseline_algo(test_grid)
    assert time_spent == 4
    
    test_grid2 = np.array([[2,2,2], [3,3,2]])
    time_spent2 = game.baseline_algo(test_grid2)
    assert time_spent2 == 8

unittest_baseline_algo()


def unittest_baseline_algo2():
    test_grid = np.array([[9,9,9,9],
                         [1,1,1,1]])
    time_spent = game.baseline_algo2(test_grid)
    assert time_spent == 13

unittest_baseline_algo2()


def unittest_get_neighbors():
    visited = []
    surrounding_pos = game.get_neighbors(test_grid, (0, 11), visited)
    assert surrounding_pos[0] == (1, 11)
    assert surrounding_pos[1] == (1, 10) #np.array([11, 1])
    assert surrounding_pos[2] == (0, 10) #np.array([11, 1])
    assert len(surrounding_pos) == 3
    
    visited = [(1, 11)]
    surrounding_pos = game.get_neighbors(test_grid, (0, 11), visited)
    assert surrounding_pos[0] == (1, 10) #np.array([11, 1])
    assert surrounding_pos[1] == (0, 10) #np.array([11, 1])
    
    assert len(surrounding_pos) == 2
    
    surrounding_pos = game.get_neighbors(test_grid, (3, 3), visited)
    assert len(surrounding_pos) == 8
    assert surrounding_pos[0] == (2, 3)
    assert surrounding_pos[1] == (2, 4)
    assert surrounding_pos[2] == (3, 4)
    assert surrounding_pos[3] == (4, 4)
    assert surrounding_pos[4] == (4, 3)
    assert surrounding_pos[5] == (4, 2)
    assert surrounding_pos[6] == (3, 2)
    assert surrounding_pos[7] == (2, 2)  
 
unittest_get_neighbors()


def unittest_get_value_of_neighbors():
    test_grid = np.array([[1,9,7,9],
                          [5,2,3,8]])
    result = game.get_value_of_neighbors(test_grid, (0, 0), [(0, 1), (1, 0)])
    assert result == [9, 5]
    
    test_grid = np.array([[1,3,6,9],
                          [2,4,6,8],
                          [1,2,3,5]])
    result = game.get_value_of_neighbors(test_grid, (1, 2), [(0, 2), (2, 2), (1, 3), (1, 1)])
    assert result == [6, 3, 8, 4]
    
unittest_get_value_of_neighbors()

def unittest_get_current_vertex():
    visited = []
    data = {"coordinate": [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
            "shortest_distance_from_starting_point": [0, 1, 99999, 99999, 99999, 99999],
            "previous_coordinate": [0,0,0,0,0,0]}
    df = pd.DataFrame(data)
    coordinate, index_pos = game.get_current_vertex(df, visited)
    assert coordinate == (0, 0)
    assert index_pos == 0
    
    visited = [(0,0)]
    coordinate, index_pos = game.get_current_vertex(df, visited)
    assert coordinate == (0, 1)
    assert index_pos == 1
    
    data = {"coordinate": [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
            "shortest_distance_from_starting_point": [2, 2, 99999, 1, 99999, 99999],
            "previous_coordinate": [0,0,0,0,0,0]}
    df = pd.DataFrame(data)
    coordinate, index_pos = game.get_current_vertex(df, visited)
    assert coordinate == (1,0)
    assert index_pos == 3

    data = {"coordinate": [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
            "shortest_distance_from_starting_point": [99999, 99999, 99999, 99999, 99999, 99999],
            "previous_coordinate": [0,0,0,0,0,0]}
    df = pd.DataFrame(data)
    coordinate, index_pos = game.get_current_vertex(df, visited)
    assert index_pos == 1
    
unittest_get_current_vertex()
 

def unittest_extract_shortest_path_from_df():
    coordinates = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    # 0 is just some arbitrary value because they are not used in the test
    shortest_distance = [0,0,0,0,0,0]
    previous_coordinate = ["NA", "NA", "NA", (0,0), (1,0), (1,1)]
    
    df = pd.DataFrame()
    df["coordinate"] = coordinates
    df["shortest_distance"] = shortest_distance
    df["previous_coordinate"] = previous_coordinate
    
    shortest_path = game.extract_shortest_path_from_df(df)
    
    assert shortest_path == [(0,0), (1,0), (1,1), (1,2)]
    
unittest_extract_shortest_path_from_df() 

# TO DO: FINISH THIS
def update_shortest_distance_and_previous_coord():
    coordinates = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    shortest_distance = [7, 4, 9, 999999, 999999, 999999]
    # not relevant in this test:
    previous_coordinate = ["NA", "NA", "NA", (0,0), (1,0), (1,1)]
    df = pd.DataFrame()
    df["coordinate"] = coordinates
    df["shortest_distance_from_starting_point"] = shortest_distance
    df["previous_coordinate"] = previous_coordinate
    
    current_vertex = (0,0)
    neighbors = [(1,0), (0, 1)]
    distance_start_to_current_neighbors = [2, 2]
        
    result = game.update_shortest_distance_and_previous_coord(neighbors, current_vertex,
                                                distance_start_to_current_neighbors, df)
    
    assert result.iloc[1, 1] == 2
    assert result.iloc[3, 1] == 2
    
  
    distance_start_to_current_neighbors = [2, 5]  
    result = game.update_shortest_distance_and_previous_coord(neighbors, current_vertex,
                                                distance_start_to_current_neighbors, df)
    
    assert result.iloc[1, 1] == 4
    assert result.iloc[3, 1] == 2
    
update_shortest_distance_and_previous_coord() 

def unittest_create_visited_unvisited():
    game = t1.Game(2, 3)
    testgrid = game.create_grid()
    visited, unvisited = game.create_visited_unvisited()
    assert len(visited) == 0
    assert len(unvisited) == testgrid.shape[0] * testgrid.shape[1]
    assert type(unvisited[0]) == tuple
    assert unvisited[0] == (0,0)
    assert unvisited[1] == (0,1)
    assert unvisited[2] == (0,2)
    assert unvisited[3] == (1,0)
    assert unvisited[4] == (1,1)
    assert unvisited[5] == (1,2)

unittest_create_visited_unvisited()

def unittest_initialize_df():
    game = t1.Game(2, 3)
    testgrid = game.create_grid()
    visited, unvisited = game.create_visited_unvisited()
    df = game.initialize_df(visited, unvisited)
    assert list(df["coordinate"]) == unvisited
    assert len(df) == len(unvisited)
    assert df.shape[1] == 3
    assert all(df.iloc[1:, 1] > 99999999) 

unittest_initialize_df()


def unittest_find_shortest_path():
    game = t1.Game(2, 3)
    testgrid = game.create_grid()
    visited = []
    unvisited = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    df = game.initialize_df(visited, unvisited)
    df = game.find_shortest_path(visited, unvisited, df)
    
unittest_find_shortest_path()
    

def unittest_get_distance_start_to_current_neighbors():
    distance_current_to_neighbors = [1, 3]
    current_vertex_row_no = 1 # i.e. the distance from star to current vertex is going to be 4
    
    coordinates = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    shortest_distance = [0, 4, 9, 999999, 999999, 999999]
    previous_coordinate = ["NA", "NA", "NA", (0,0), (1,0), (1,1)]
    
    df = pd.DataFrame()
    df["coordinate"] = coordinates
    df["shortest_distance_from_starting_point"] = shortest_distance
    df["previous_coordinate"] = previous_coordinate
    
    result = game.get_distance_start_to_current_neighbors(distance_current_to_neighbors, df, current_vertex_row_no)
    assert len(result) == 2
    assert 5 in result
    assert 7 in result
   
unittest_get_distance_start_to_current_neighbors()
    
def find_shortest_path():
    visited = []
    game = t1.Game(height, width)
    game.grid_values = np.array([[1, 9, 6, 1],
                                 [1, 9, 9, 1],
                                 [9, 9, 9, 5]])
    game.grid_distances = np.matrix(np.ones((game.grid_values.shape[0], game.grid_values.shape[1])) * np.inf)
    game.grid_distances[0,0] = game.grid_values[0,0]
    game.end_point = (game.grid_values.shape[0]-1, game.grid_values.shape[1]-1)
    game.find_shortest_path(visited)
    time_spent = game.grid_distances[-1, -1]
    assert time_spent == 22
    
    visited = []
    game.grid_values = np.array([[1, 9, 9, 9],
                                 [9, 1, 1, 9],
                                 [9, 9, 9, 5]])
    game.grid_distances = np.matrix(np.ones((game.grid_values.shape[0], game.grid_values.shape[1])) * np.inf)
    game.grid_distances[0,0] = game.grid_values[0,0]
    game.end_point = (game.grid_values.shape[0]-1, game.grid_values.shape[1]-1)
    game.find_shortest_path(visited)
    time_spent = game.grid_distances[-1, -1]
    assert time_spent == 8
    
    visited = []
    game.grid_values = np.array([[1, 9, 9, 9, 9, 9, 9, 9],
                                 [9, 9, 9, 9, 9, 9, 9, 9],
                                 [9, 9, 9, 9, 9, 9, 9, 9],
                                 [9, 9, 9, 9, 9, 9, 9, 9],
                                 [1, 9, 9, 9, 9, 9, 9, 9],
                                 [1, 1, 1, 1, 1, 1, 1, 5]]) # 43
    game.grid_distances = np.matrix(np.ones((game.grid_values.shape[0], game.grid_values.shape[1])) * np.inf)
    game.grid_distances[0,0] = game.grid_values[0,0]
    game.end_point = (game.grid_values.shape[0]-1, game.grid_values.shape[1]-1)
    game.find_shortest_path(visited)
    time_spent = game.grid_distances[-1, -1]
    assert time_spent == 40
    
    visited = []
    game.grid_values = np.array([[1, 9, 9, 9, 9, 9, 9, 9],
                                 [9, 9, 9, 9, 9, 9, 9, 9],
                                 [9, 9, 9, 9, 9, 9, 9, 9],
                                 [9, 9, 9, 9, 9, 9, 9, 9],
                                 [9, 9, 9, 9, 9, 9, 9, 9],
                                 [1, 9, 9, 9, 9, 9, 9, 9],
                                 [1, 9, 9, 9, 9, 9, 9, 9],
                                 [1, 9, 9, 9, 9, 9, 9, 9],
                                 [1, 1, 1, 1, 1, 1, 1, 5]]) # 43
    game.grid_distances = np.matrix(np.ones((game.grid_values.shape[0], game.grid_values.shape[1])) * np.inf)
    game.grid_distances[0,0] = game.grid_values[0,0]
    game.end_point = (game.grid_values.shape[0]-1, game.grid_values.shape[1]-1)
    game.find_shortest_path(visited)
    time_spent = game.grid_distances[-1, -1]
    assert time_spent == 51
    
find_shortest_path()
    
    
    
def unittest_extract_path_from_parent_dict():
    visited = []
    game = t1.Game(height, width)
    game.grid_values = np.array([[1, 9, 6, 1],
                                 [1, 9, 9, 1],
                                 [9, 9, 9, 5]])
    game.grid_distances = np.matrix(np.ones((game.grid_values.shape[0], game.grid_values.shape[1])) * np.inf)
    game.grid_distances[0,0] = game.grid_values[0,0]
    game.end_point = (game.grid_values.shape[0]-1, game.grid_values.shape[1]-1)
    game.find_shortest_path(visited)
    time_spent = game.grid_distances[-1, -1]
    assert time_spent == 22
    
shortest_path = game.extract_shortest_path()
    
'''
def unittest_get_next_node():
    grid = np.array([[1, 9, 6, 1],
                     [1, 9, 9, 1],
                     [9, 9, 9, 5]])
    
    # In reality, the grid and the pheromone matrix should not be the same, 
    # but it does not matter for the purposes of this test
    ant = t1.ACO(grid, 0.5)
    ant.pheromone_matrix = grid
    
    neighbors = [(0,1), (1,0)]
    next_node = ant.get_next_node(neighbors)
    
    # Method should return the neighbor with the highest amount of pheromone
    assert next_node == (0,1)
    
unittest_get_next_node()
    '''
    
    
############################ TESTING ACO #####################################
def unittest_update_pheromones():
    arr = np.array([[1.0,2.0,3.0],
                [4.0,5.0,6.0]])
    
    # In reality, the grid and the pheromone matrix should not be the same, 
    # but it does not matter for the purposes of this test
    ant = t1.ACO(arr, 0.5)
    ant.pheromone_matrix = arr
    path_length = 2
    path = [(0,0), (0,1)]
    ant.rho = 0.1
    
    ant.update_pheromone_matrix(path, path_length)
    
    assert ant.pheromone_matrix[path[0]] == 1.45
    assert ant.pheromone_matrix[path[1]] == 2.45
    
    #arr[tuple(path)] = (1-rho)*(1/path_length)+arr[tuple(path)]
unittest_update_pheromones()


def unittest_get_pheromone_of_neighbors():
    pheromone_matrix = np.array([[1, 9, 60, 10],
                                 [1, 90, 90, 10],
                                 [90, 90, 90, 50]])
    
    ant = t1.ACO(pheromone_matrix, 0.5)
    ant.pheromone_matrix = pheromone_matrix
    
    neighbors = [(0,0), (1,0)]
    
    res = ant.get_pheromone_of_neighbors(neighbors)
    
    assert res[1] == 1
    assert res[0] == 1
    
    neighbors = [(0,1), (1,0), (2,1), (1,2)]

    res = ant.get_pheromone_of_neighbors(neighbors)
    assert res[0] == 9
    assert res[1] == 1
    assert res[2] == 90
    assert res[3] == 90
    
unittest_get_pheromone_of_neighbors()
    
def unittest_get_dist_current_to_neighbors():
    grid = np.array([[1, 9, 60, 10],
                     [1, 90, 90, 10],
                     [90, 90, 90, 50]])
    
    ant = t1.ACO(grid, 0.5)
    ant.grid = grid
    
    
    current = (0,0)
    neighbors = [(0,1), (1,0)]
    res = ant.get_dist_current_to_neighbors(current, neighbors)
    assert res[0] == 10
    assert res[1] == 2
    
    current = (1,1)
    neighbors = [(0,1), (1,0), (2,1), (1,2)]

    res = ant.get_dist_current_to_neighbors(current, neighbors)
    assert res[0] == 99
    assert res[1] == 91
    assert res[2] == 180
    assert res[3] == 180

unittest_get_dist_current_to_neighbors()

def unittest_get_probabilities():
    grid = np.array([[1, 9, 60, 10],
                     [1, 90, 90, 10],
                     [90, 90, 90, 50]])
    
    pheromone_matrix = np.array([[1, 1, 60, 10],
                                 [9, 90, 90, 10],
                                 [90, 90, 90, 50]])
    
    
    ant = t1.ACO(grid, 0.5)
    ant.grid = grid
    ant.pheromone_matrix = pheromone_matrix
    
    neighbors = [(1,0), (0,1)]
    current = (0,0)
    
    res = ant.get_probabilities(neighbors, current)
    
    assert math.isclose(1.0, np.sum(res), rel_tol=0.001)
    assert res[0] == 0.9782608695652175
    assert len(res) == len(neighbors)
    
    neighbors = [(0,1), (1,0), (2,1), (1,2)]
    current = (1,1)
    
    res = ant.get_probabilities(neighbors, current)
    
    assert math.isclose(1.0, np.sum(res), rel_tol=0.001)
    assert res[2] == res[3]
    assert len(res) == len(neighbors)
    
    
    grid = np.array([[1, 9, 60, 10],
                     [1, 90, 90, 10],
                     [90, 90, 90, 50]])
    
    pheromone_matrix = np.array([[0., 0., 60, 10],
                                 [0., 90, 90, 10],
                                 [90, 90, 90, 50]])
    
    
    ant = t1.ACO(grid, 0.5)
    ant.grid = grid
    ant.pheromone_matrix = pheromone_matrix
    
    neighbors = [(1,0), (0,1)]
    current = (0,0)
    res = ant.get_probabilities(neighbors, current)
    
unittest_get_probabilities()   


def unittest_get_next_node():
    grid = np.array([[1, 90000, 60, 10],
                     [1, 90, 90, 10],
                     [90, 90, 90, 50]])
    
    pheromone_matrix = np.array([[1, 1, 60, 10],
                                 [90000, 90, 90, 10],
                                 [90, 90, 90, 50]])
    
    ant = t1.ACO(grid, 0.5)
    ant.grid = grid
    ant.pheromone_matrix = pheromone_matrix
    
    current = (0,0)
    neighbors = [(1,0), (0,1)]
    
    res = []
    
    for i in range(100):
        res.append(ant.get_next_node(neighbors, current))
        
    assert mode(res) == (1,0)
    
    grid = np.array([[1, 1, 60, 10],
                     [1, 90, 90, 10],
                     [90, 90, 90, 50]])
    
    pheromone_matrix = np.array([[1, 1, 60, 10],
                                 [2, 90, 90, 10],
                                 [90, 90, 90, 50]])
    
    ant = t1.ACO(grid, 0.5)
    ant.grid = grid
    ant.pheromone_matrix = pheromone_matrix
        
    res = []
    
    for i in range(100):
        res.append(ant.get_next_node(neighbors, current))
        
    assert mode(res) == (1,0)
    
    grid = np.array([[1, 90, 60, 10],
                     [5, 90, 90, 10],
                     [90, 90, 90, 50]])
    
    pheromone_matrix = np.array([[1, 90, 60, 10],
                                 [1000, 90, 90, 10],
                                 [80, 90, 90, 50]])
    
    ant = t1.ACO(grid, 0.5)
    ant.grid = grid
    ant.pheromone_matrix = pheromone_matrix
    current = (1,1)
    neighbors = [(0,1), (1,0), (2,1), (1,2)]
    
    res = []
    
    for i in range(100):
        res.append(ant.get_next_node(neighbors, current))
        
    assert mode(res) == (1,0)
    
    grid = np.array([[1, 90, 60, 10],
                     [5, 90, 90, 10],
                     [90, 90, 90, 50]])
    
    pheromone_matrix = np.array([[1, 90, 60, 10],
                                 [1000, 90, 90, 10],
                                 [80, 90, 90, 50]])
    
    ant = t1.ACO(grid, 0.5, 5, 5)
    ant.grid = grid
    ant.pheromone_matrix = pheromone_matrix
    current = (1,1)
    neighbors = [(0,1), (1,0), (2,1), (1,2)]
    
    res = []
    
    for i in range(100):
        res.append(ant.get_next_node(neighbors, current))
        
    assert mode(res) == (1,0)
    
unittest_get_next_node()

    
def unittest_get_path_length():
    grid = np.array([[1, 90, 60, 10],
                     [5, 90, 7, 10],
                     [90, 90, 90, 50]])
    
    
    ant = t1.ACO(grid, 0.5, 5, 5)
    ant.grid = grid
    
    path = [(0,0), (1,0), (2,0)]
    
    res = ant.get_path_lenght(path)
    
    assert res == 96
    
    path = [(0,0), (0,1), (0,2), (1,2)]
    
    res = ant.get_path_lenght(path)
    assert res == 158
    
    path = []
    
    res = ant.get_path_lenght(path)
    assert res == 0
    
unittest_get_path_length()





    




