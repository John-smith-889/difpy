"""
Created on Thu Dec 19 22:29:52 2019


    Module enables computing score of information spreding capability
    for each node, and check correlations between this computed score 
    and other features associated with nodes.
    

    Objects
    ----------
    nodes_score_simulation : function
        A function computes score of information spreding capability
        for each node.

    
    feature_importance : function
        A function computes correlations between nodes' scores variable 
        and nodes features.
    

"""


import difpy as dp
import time
import networkx as nx

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

#=============================================================================#
# nodes_score function #
#======================#

def nodes_score_simulation(
        G,
        log_info_interval = None, # interval of information log 
               
        n = 5, # number of simulation steps in simulation
        sequence_len = 10, # number of simulations in one sequence
        
        kernel = 'weights', # kernel type
        custom_kernel = None, # custom kernel function
        WERE_multiplier = 10, 
        oblivion = False, # information oblivion feature 
        engagement_enforcement = 1.00
        ): 
                
    
    """ Compute nodes information propagation capability.
    
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
    
    log_info_interval: integer, optional
        Interval between iterations to log simulations information 
        in the console. If None, information is hidden.
    
        
        
    Parameters wrapped from simulation_sequence function:
    -----------------------------------------------------
    
    sequence_len : integer
        A number of simulations to perform in one sequence.
        
    
    
    Parameters wrapped from simulation function:
    --------------------------------------------------
    
    n : integer
        A number of simulation steps for a given graph.
        
    
    
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
    list_solution : list
        List with nodes information propagation capabilities. Each value 
        is an average increment of aware nodes per one simulation step
        for each node.
        
    
    """        

    # Start time measuring
    start = time.time()

    # Compute number of nodes
    population = range(len(G))

    # Create lists for saving score 
    new_solution = 0
    list_solution = []

    #====================================#
    # General loop for solutions testing #
    #====================================#
    
    for i in population:
    
        # Add 'unaware' state for all nodes
        nx.set_node_attributes(G, 'unaware', 'state') # (G, value, key)
        
        # Set choosen node as aware
        G.nodes[i]['state'] = 'aware'
        
        # perform sequence of simulations
        new_solution = dp.simulation_sequence(G,
                                              n,
                                              sequence_len,
                                              kernel,
                                              custom_kernel,
                                              WERE_multiplier,
                                              oblivion,
                                              engagement_enforcement
                                              )
              
        # Save new node result to list
        list_solution.append(new_solution)
        
        # Show log information
        if log_info_interval is not None:
            if i > 0:
                if i % log_info_interval == 0:
                    end = time.time()
                    print(i, "Iterations passed",\
                          "in", \
                          round(end - start, 2), "seconds." )
    #==============#
    # Show results #
    #==============#
    
    print("")                
    print("List of solutions:",\
          list_solution)
    #print("")
    #print("Set of initial aware nodes:", best_solution[1] )
        
    return list_solution



#=============================================================================#
# feature_importance function #
#=============================#

def feature_importance(
        G, # NetworkX graph
        X, # Nodes attributes 
        show = True, # Show features' performances
        log_info_interval = 1, # interval of information log 
                       
        sequence_len = 200, # number of simulations in one sequence
        n = 10, # number of simulation steps simulation
                       
        kernel = 'weights', # kernel type
        custom_kernel = None, # custom kernel function
        WERE_multiplier = 10, 
        oblivion = False, # information oblivion feature 
        engagement_enforcement = 1.00
        ):

    
    """ Compute importace of features associated with nodes. It shows 
    importance of features in context of information propagation
    capability of nodes.
    
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
        
    X: ndarray
        Ndarray with nodes' features which importance will be investigated.
        
    log_info_interval: integer, optional
        Interval between iterations to log simulations information 
        in the console. If None, information is hidden.
    
        
        
    Parameters wrapped from simulation_sequence function:
    -----------------------------------------------------
    
    sequence_len : integer
        A number of simulations to perform in one sequence.
        
    
    
    Parameters wrapped from simulation_steps function:
    --------------------------------------------------
    
    n : integer
        A number of simulation steps for a given graph.
        
    
    
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
    feature_importances : ndarray
        Ndarray with feature importance for each variable.
        Feature importance is computed with xgboost package.
        
    
    """        

    #=========================#
    # Compute score for nodes #
    #=========================#
    
    Y = nodes_score_simulation(G,
                    log_info_interval, # interval of information log 
                       
                    n, # number of simulation steps in simulation
                    sequence_len, # number of simulations in one sequence
                       
                    kernel, # kernel type
                    custom_kernel, # custom kernel function
                    WERE_multiplier, 
                    oblivion, # information oblivion feature 
                    engagement_enforcement
                    )

    X_train, X_test, Y_train, Y_test \
    = train_test_split(X, Y, test_size = 0.20, random_state = 10)

    
    # Data modelling
    model_01 = XGBRegressor(objective='reg:squarederror')
    model_01.fit(X_train, Y_train)


    # feature importance
    feature_importances = model_01.feature_importances_
    
    # Model evaluation 
    # ---------
    
    if show == True:
        print("")
        print("Feature importances:")
        print("")
        for i, i2 in enumerate(feature_importances):
            print("Variable", i+1, ":", i2)

    return feature_importances
