# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:36:07 2020

@author: groes
"""



import random 
import numpy as np
import copy

class Game:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.starting_point = (0, 0)
        self.grid = self.create_grid()
        self.end_point = (self.grid.shape[0]-1, self.grid.shape[1]-1)
        
    def create_grid(self):
        return np.random.randint(0, 9, (self.height, self.width))
    
         
  
class Algorithm:
    
    def __init__(self, grid):
        self.grid = grid
        self.height = grid.shape[0]
        self.width = grid.shape[1]
        self.starting_point = (0,0)
        self.end_point = (grid.shape[0]-1, grid.shape[1]-1)
        # Can I reference self.heigh and self.width already here?
        self.all_vertices = set((i,j) for i in range(self.height) for j in range(self.width))
        
        
    def get_neighbors(self, current_vertex):
        """
        Creating a set of neighbors. This method does not consider whether the
        neighbors are out of bounds as this is handled with a look-up in the set
        all_vertices for speed efficiency inside the for loop which was already
        in place in find_shortest_path

        Parameters
        ----------
        current_vertex : TYPE
            DESCRIPTION.

        Returns
        -------
        neighbors : SET
            Set containing neighbors of current vertex

        """
        i = current_vertex[0]
        j = current_vertex[1]
        
        neighbors = {(i-1, j), (i, j+1), (i+1, j), (i, j-1)}
        # Changing code so that I use set instead of list for computational efficiency 
        return neighbors
    
    # Implementing as method so that its testable
    def get_value_of_neighbors(self, neighbors):
    
        '''
        
        Parameters
        ----------
    
        neighbors : TYPE
            A list containing tuples. Each tuple represent a cell adjacent
            to (the current) position
    
        Returns
        -------
        distances : LIST
            A list containing integers that represent the distance from position
            (or "vertex" in graph lingo) to each of its surrounding positions 
            (or "vertices"). The ith integer in distances corresponds to the 
            distance from position to the ith position in neighbors.
    
        '''
        value_of_neighbors = [self.grid[neighbor] for neighbor in neighbors]
        
        return value_of_neighbors
    
        
class RandomAlgo(Algorithm):
    def __init__(self, grid):
        super().__init__(grid)
        
    def run_random_algo(self):
        current_position = (0,0)
        time_spent = self.grid[current_position]
        path = [current_position]
        
        while self.end_point != current_position:
            neighbors_unfiltered = self.get_neighbors(current_position) 
            neighbors = []
            
            for neighbor in neighbors_unfiltered:
                if neighbor in self.all_vertices:
                    neighbors.append(neighbor)
            
            # Setting next node to visit:
            current_position = random.choice(neighbors)

            path.append(current_position)
            time_spent += self.grid[current_position]
            
        return path, time_spent
            

class BaselineAlgo(Algorithm):
    """
    This algo always either moves right or down. If the next cell to the
    right is lower than the next cell below, it moves right and vice versa
        """
    def __init__(self, grid):
        super().__init__(grid)
        
    def run_baseline(self):
        current_position = (0,0)
        time_spent = self.grid[current_position]
        
        path = [current_position]
        while self.end_point != current_position:
            neighbors_unfiltered = [(current_position[0]+1, current_position[1]),
                         (current_position[0], current_position[1]+1)]
            
            neighbors = []
            for neighbor in neighbors_unfiltered:
                if neighbor in self.all_vertices:
                    neighbors.append(neighbor)
                    
            distance_to_neighbors = []
            for neighbor in neighbors:
                distance_to_neighbors.append(self.grid[neighbor])
                
            idx_shortest_dist = distance_to_neighbors.index(min(distance_to_neighbors))
            
            current_position = neighbors[idx_shortest_dist]
            
            path.append(current_position)
            time_spent += self.grid[current_position]
        
        return path, time_spent
        
        
class Dijkstras(Algorithm):
    
    def __init__(self, grid):
        super().__init__(grid)


    def run_dijkstras(self):
        """
         Wrapper method for all other Dijkstra's methods.

        Returns
        -------
        path : list
            List containing the (i,j) positions that constitute the shortest
            path from start to end.
            
        time_spent : int
            The sum of the values of the visited vertices

        """
        
        # Each cell in this grid represents the distance from start to the cell
        self.grid_distances = np.matrix(np.ones((self.height, self.width)) * np.inf) #np.zeros((self.height, self.width))
        
        # Each cell in this grid represent the value of the cell, i.e. the distance
        # from [0,0] to [0,1] is the value of [0,0] + the value of [0,1]
        self.grid_values = copy.copy(self.grid)
        
        # Setting the value of the first cell
        self.grid_distances[0,0] = self.grid_values[0,0]

        self.path, self.time_spent = self.find_shortest_path() # get_shortest_path
        
        
        return self.path, self.time_spent
        

    def extract_path_from_parent_dict(self):
        """
        From a dictionary, parent_nodes, that contains the parent nodes of 
        each of the visited nodes, the shortest path from end node to start node
        is extracted. 
        
        The dictionary parent_nodes is created in find_shortest_path() 

        Returns
        -------
        path : LIST
            List containing the (i,j) positions that constitute the shortest
            path from start to end

        """
    
        path = [self.end_point]
        keep_going = True
        vertex = self.end_point
        iteration = 1
        while keep_going:
            iteration += 1
            parent = self.parent_nodes[vertex]  
            path.append(parent)
            vertex = parent
            if vertex == (0,0):
                keep_going = False
        path.reverse()
        return path
             
        
    def find_shortest_path(self): 
        """
        This method finds the shortest path from start to end. By means of
        extract_path_from_parent_dict() constructs a list representing the 
        shortest path. It also determines the time spent going from start to
        end via the shortest path.

        Returns
        -------
        path : LIST
            List containing the (i,j) vertex that constitute the shortest
            path from start to end
        time_spent : INT
            Integer representing the sum of the values of the vertices visited
            when going from start to end via the shortest path.

        """
        # For efficiency, I will store visited vertices so as to not visit them again
        visited = []
        
        # Initializing dictionary that will contain the parent node of each visited node
        self.parent_nodes = {self.starting_point : self.starting_point}
        
        # The next vertex the algorithm will visit will always be the vertex
        # from queue_vertices with the lowest value (i.e. the shortest distance
        # from the starting point)
        self.queue_vertices = [self.starting_point] 
        self.queue_values = [self.grid_values[self.starting_point]]
        iteration = 1

        while self.queue_vertices:
            
            # Finding the min value of queue_values so that I can find its 
            # index position and use its index position to get the current_vertex
            value_of_current = min(self.queue_values)
            current_vertex_index = self.queue_values.index(value_of_current)
            current_vertex = self.queue_vertices[current_vertex_index]

            iteration += 1

            # Getting the (i,j) of neighbors so that I can find their values
            neighbors = self.get_neighbors(current_vertex)
        
            for neighbor in neighbors:
                
                if neighbor in self.all_vertices and neighbor not in visited:
                    value_of_neighbor = self.grid[neighbor]
                    distance_start_to_neighbor = value_of_current + value_of_neighbor
                    
                    if neighbor not in self.queue_vertices:
                        self.queue_vertices.append(neighbor)
                        self.queue_values.append(distance_start_to_neighbor)
                        
                    # Updating grid_distances and parent_nodes if the distance from the 
                    # current vertex (which is (0,0) in first iteration) to its neighbors
                    # are smaller than the distances already registered
                    if distance_start_to_neighbor < self.grid_distances[neighbor]:
                        self.grid_distances[neighbor] = distance_start_to_neighbor
                        self.parent_nodes[neighbor] = current_vertex
                        self.update_queue(neighbor, distance_start_to_neighbor)
                        
                # Skip iteration if neighbor is out of bounds or have been visisted    
                else:
                    continue
                    
            visited.append(current_vertex)
            del self.queue_vertices[current_vertex_index]
            del self.queue_values[current_vertex_index]
            
        
        path = self.extract_path_from_parent_dict()#construct_path()
        time_spent = self.grid_distances[-1, -1]
        
        return path, time_spent

    def update_queue(self, neighbor, distance_start_to_neighbor):
        node_index = self.queue_vertices.index(neighbor)
        self.queue_values[node_index] = distance_start_to_neighbor
        
         
     
class ACO(Algorithm):
    # No code is shown in this video, but I used it to get an idea of what ACO is:
    # https://www.youtube.com/watch?v=783ZtAF4j5g&list=PLDANVeKlceVTWBGi99VhGJKpwVoXPS-4c&index=24&t=635s&ab_channel=AliMirjalili 
    
    def __init__(self, grid, initial_pheromone_level=1):
        # alpha and beta are used to calculate the probabilities of the neighbors
        super().__init__(grid)
        self.all_vertices = [(i,j) for i in range(self.height) for j in range(self.width)] #set((i,j) for i in range(self.height) for j in range(self.width)) # #
        self.initial_level = initial_pheromone_level
        self.pheromone_matrix = np.ones((self.height, self.width))
    
    def get_pheromone_of_neighbors(self, neighbors):
        pheromone_neighbors = []
        for neighbor in neighbors:
            pheromone_neighbors.append(self.pheromone_matrix[neighbor])
        return pheromone_neighbors
    
    def get_dist_current_to_neighbors(self, current, neighbors):
        value_of_current = self.grid[current]
        #distances_current_to_neighbors = value_of_current + self.grid[tuple(neighbors)]
        distances_current_to_neighbors = []
        for neighbor in neighbors:
            #print("neighbor printed from inside get_dist_current_to_neighbors")
            #print(neighbor)
            distances_current_to_neighbors.append(value_of_current + self.grid[neighbor])
        
        return distances_current_to_neighbors
    
    
    def get_probabilities(self, neighbors, current):
        
        pheromone_of_neighbors = self.get_pheromone_of_neighbors(neighbors)

        
        distance_to_neighbors = self.get_dist_current_to_neighbors(current, neighbors)

        
        numerators = []
        for i in range(len(neighbors)):
            # Using the "quality" of the edge, (1/distance_to_neighbors[i]), to determine probability 
            distance_to_neighbor = distance_to_neighbors[i]
            if distance_to_neighbor == 0:
                # Setting value to avoid dividing by 0
                distance_to_neighbor = 0.5
            numerators.append((pheromone_of_neighbors[i]**self.alpha) * ((1/distance_to_neighbor)**self.beta)) 
  
        
        denominator = np.sum(numerators)

        
        probabilities = []
        for numerator in numerators:
            probabilities.append(numerator/denominator)

        return probabilities

        
    def get_next_node(self, neighbors, current):
        probabilities = self.get_probabilities(neighbors, current)
        cumulative_sum = np.cumsum(probabilities[::-1])[::-1] 
        random_number = random.random()
        
        for i in range(len(cumulative_sum)-1):
            if cumulative_sum[i+1] < random_number <= cumulative_sum[i]:
                return(neighbors[i])

        return neighbors[-1]

    def run_ACO(self, ants, rho=0.5, alpha=0.5, beta=0.5):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        
        time_since_improvement = 0
    
        keep_going_little_ant = True
    
        shortest_path = None
        length_shortest_path = 9999999999999999999999
    
        for ant in range(ants):
            current_node = (0,0)
            path = []
            print("Sending in ant number {}".format(ant))
            time_spent = 0
            # Per ant
            while keep_going_little_ant:
                path.append(current_node)
                time_spent += self.grid[current_node]
                unfiltered_neighbors = self.get_neighbors(current_node)
                
                # The ant can only consider nodes that exist and that it has not visited 
                neighbors = []
                for neighbor in unfiltered_neighbors:
                    if neighbor in self.all_vertices and neighbor not in path:
                        neighbors.append(neighbor)
                
                # If the ant ends up surrounded by nodes it has already visited,
                # the ant goes back to where it started
                if len(neighbors) == 0:
                    for neighbor in unfiltered_neighbors:
                        if neighbor in self.all_vertices:
                            neighbors.append(neighbor)
                
                # If one of the surrounding vertices is the end point, exit while loop
                if self.end_point in neighbors:
                    path.append(self.end_point)
                    time_spent += self.grid[self.end_point]
                    break
                    
                # Setting next node
                current_node = self.get_next_node(neighbors, current_node)
            
            # To avoid dividing by zero when updating pheromones
            if time_spent > 0:
                self.update_pheromone_matrix(path, time_spent)
            
            if time_spent < length_shortest_path:
                length_shortest_path = time_spent
                shortest_path = path
                print("Found shorter path")
            else:
                time_since_improvement += 1
                print("Time since last improvement: {}".format(time_since_improvement))
        
        if self.end_point not in shortest_path:
            print("Something went wrong because end point is not in shortest path found")
            return None, None
        
        return shortest_path, length_shortest_path
        
    
    def update_pheromone_matrix(self, path, path_length):
        update = 1/path_length
        
        for i in path: 
            self.pheromone_matrix[i] += update
        
        self.pheromone_matrix * (1-self.rho)

    def get_path_lenght(self, path):
        try:
            path_length = 0 #np.sum(self.grid[tuple(path)])
            for node in path:
                path_length += self.grid[node]
            return path_length
        except:
            print("Path has length 0")
            return 0
            
