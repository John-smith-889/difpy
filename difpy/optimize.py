"""
Created on Tue Dec 17 01:12:39 2019


    Module enables optimization of best nodes set for information 
    seeding in Difpy package.
    

    Objects
    ----------
    optimize_centrality() : function
        A function computes closeness centrality with given networkx function 
        for choosen set of nodes.

    
    optimize_rs() : function
        A function searches for best set of nodes for information diffusion
        with random search method. 
        
    
"""

import difpy as dp
import networkx as nx
#import numpy as np
# import random # used only by difpy subfunction
#import matplotlib.pyplot as plt
#import copy
import time
import random


#=============================================================================#
# optimize_centrality #
#=====================#

def optimize_centrality(G,
                        number_of_nodes = 1,
                        distance = None,
                        wf_improved = True):

    """ Show n best nodes for information diffusion in a graph. 
    Betweeness centrality method from networkx package is used.
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
    
    number_of_nodes : integer
        Number of best nodes to show.
        
    
    Parameters wrapped from closeness_centrality networkx function:
    ---------------------------------------------------------------
    
    distance : string, optional
        Name of the edges attribute used as distance measure.
    
    wf_improved : bool
        Logic value confirms to use improved version of an algorithm.

                   
    Returns
    -------
    n_best_nodes : list of tuples
        List of tuples with nodes' numbers/names and centrality values.
        
    
    """        
    
    # Calculate closeness centrality measure for nodes
    closeness_centrality = nx.closeness_centrality(G,
                                                   distance = distance, # this arg is not positional
                                                   wf_improved = wf_improved)
    # Sorting nodes centrality descending
    closeness_centrality = sorted(closeness_centrality.items(), 
                                  key=lambda x: x[1], reverse=True)
    closeness_centrality

    # Choose n best nodes
    n_best_nodes = []
    n_best_nodes = closeness_centrality[0:number_of_nodes]

    return n_best_nodes




#=============================================================================#
# optimize_rs #
#=============#
    

def optimize_rs(G,
                number_of_nodes, # number of nodes to seed
                number_of_iter, # number of iterations 
                log_info_interval = None, # interval of information log 
               
                n = 5, # number of simulation steps simulation
                sequence_len = 10, # number of simulations in one sequence
                
                kernel = 'weights', # kernel type
                custom_kernel = None, # custom kernel function
                WERE_multiplier = 10, 
                oblivion = False, # information oblivion feature 
                engagement_enforcement = 1.00
                ): 
                
    """ Show n best nodes for information diffusion in a graph. 
    Random search method is used to optimization.
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
        
    number_of_nodes: integer
        Number of nodes we want to choose to seed information 
        among population.
    
    number_of_iter: integer
        Number of iterations of random search to perform.
    
    log_info_interval: integer, optional
        Interval between iterations to log simulations information 
        in the console. If None, information is hidden.
    
        
        
    Parameters wrapped from simulation_sequence function:
    -----------------------------------------------------
    
    n : integer
        A number of simulation steps for a given graph.
    
    sequence_len : integer
        A number of simulations to perform in one sequence.
        
    
    
    Parameters wrapped from simulation_step function:
    -------------------------------------------------
          
    kernel : string
        Levels: "weights", "WERE", "custom"
        
        * weights - means that probability of information propagation is 
            equals to bond value between actors
        * WERE - probability of information propagation equals 
            Weights-extraversion-receptiveness-engagement equation
        * custom - probability of information propagation is computed 
            with custom function
            
    engagement_enforcement : float
        Enforcement of agent engagement by multiplier. 
        If engagement_enforcement == 1, no enforcement occurs.
        
        1) Agent enforce its engagement after oblivion, 
            (later its easier to internalize information again for this
            agent, at least with WERE kernel)
        2) Agent A enforce its engagement during information diffusion step,
            when another agent B is trying to pass information towards 
            agent A, but agent A is already aware.
            
    custom_kernel : function
        Function which compute probability of information propagation
        for each node in simulation step.
    
    WERE_multiplier : Float, optional
        Multiplier used for scaling WERE kernel outcome.
    
    oblivion : bool, optional
        Option which enable agents information oblivion. 
        

        
    Returns
    -------
    n_best_nodes : list of tuples
        List of tuples with nodes' numbers/names and centrality values.
        
    
    """        

    # Start time measuring
    start = time.time()

    # Compute number of nodes
    population = range(len(G))

    # Create lists for saving score 
    candidate_solution = 0
    best_solution = [0,[0]]

    #====================================#
    # General loop for solutions testing #
    #====================================#
    
    for i in range(number_of_iter):
    
        # Add 'unaware' state for all nodes
        nx.set_node_attributes(G, 'unaware', 'state') # (G, value, key)
        
        # Sample choosen 
        infected_agents_id = random.sample(population, number_of_nodes)
        infected_agents_id

        # Set those nodes as aware
        for v in infected_agents_id:
            G.nodes[v]['state'] = 'aware'
        
        # perform sequence of simulations
        candidate_solution = dp.simulation_sequence(G,
                                                    n,
                                                    sequence_len,
                                                    kernel,
                                                    custom_kernel,
                                                    WERE_multiplier,
                                                    oblivion,
                                                    engagement_enforcement
                                                    )
              
        # Save results if its better than before    
        if candidate_solution > best_solution[0]:
            best_solution[0] = candidate_solution
            best_solution[1] = infected_agents_id

        # Show log information
        if log_info_interval is not None:
            if i > 1:
                if i % log_info_interval == 0:
                    end = time.time()
                    print(i, "Iterations passed with best solution:",\
                          round(best_solution[0],4), "in", \
                          round(end - start, 2), "seconds." )
    #==============#
    # Show results #
    #==============#
    
    print("")                
    print("Best aware agents increment per simulation step:",\
          best_solution[0])
    print("")
    print("Set of initial aware nodes:", best_solution[1] )
        
    return best_solution

