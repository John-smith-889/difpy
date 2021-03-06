"""
Created on Thu Dec 12 01:02:53 2019


    Module enables simulations in Difpy package on NetworkX graphs.

    There are possibilities to perform singular simulation step,
    whole simulation or simulation sequence. With those functions 
    we may investigate how informations spread in certain graph.
    In simulation step function, and simulation function we get graph's
    nodes's properties fingerprints from all steps of simulation
    as an output. In simulation sequence function we get statistics of 
    information spreading as an output.
    

    Objects
    ----------
    simulation_step : function
        A function performs one step of simulation.

    
    simulation : function
        A function performs one simulation with multiple steps.
       
    
    simulation_sequence : function
        A function performs one sequence of simulations.


"""

import difpy as dp
# import networkx as nx # used only by difpy subfunction
import numpy as np
# import random # used only by difpy subfunction
import matplotlib.pyplot as plt
import copy

#=============================================================================#
# Function one simulation step #
#==============================#

def simulation_step(G, # NetworkX graph
                    pos = None,
                    kernel = 'weights',
                    engagement_enforcement = 1.00,
                    custom_kernel = None,
                    WERE_multiplier = 10, 
                    oblivion = False, 
                    draw = False, 
                    show_attr = False):

    """ Perform one simulation step of information diffusion 
        in a graph G.
    
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
        
        To use default WERE information propagation kernel, nodes of G 
        need to have extraversion, receptiveness, engagement parameters.
        
    pos : dictionary with 2 element ndarrays as values
       Object contains positions of nodes in the graph chart. Pos is used 
       to draw the graph after simulation step.
    
    kernel : string
        Levels: "weights", "WERE", "custom"
        
        * weights - means that probability of information propagation is 
            equals to bond value between actors
        * WERE - probability of information propagation equals 
            Weights-extraversion-receptiveness-engagement equation
        * custom - probability of information propagation is computed 
            with custom function
            
    engagement_enforcement : float
        Reinforcement of agent engagement by multiplier. 
        If engagement_enforcement == 1, no reinforcement occurs.
        
        1) Agent reinforce its engagement after oblivion, 
            (later its easier to internalize information again for this
            agent, at least with WERE kernel)
        2) Agent A reinforce its engagement during information diffusion step,
            when another agent B is trying to pass information towards 
            agent A, but agent A is already aware.
            
    custom_kernel : function
        Function which compute probability of information propagation
        for each node in simulation step.
    
    WERE_multiplier : Float, optional
        Multiplier used for scaling WERE kernel outcome.
    
    oblivion : bool, optional
        Option which enable agents information oblivion. 
        
    draw : bool, optional
        Draw graph.


    Returns
    -------
    G : graph
        A modified networkx graph object is returned after simulation.


    """        

    for n in G.nodes():
    
        
        #=================#
        # Oblivion option #
        #=================#
        
        #  Oblivion and increasing engagement
        
        if oblivion == True:
            
            if G.nodes[n]['state'] == 'aware':

                # Calculate oblivion_probability for certain node (more aware neighbours - lower oblivion)
                # oblivion_prob - is random uniform, and
                # dependent on what percent of neighbour are aware
                
        
                aware = [d['state'] for i,d in G.nodes.data() if i in list(G.neighbors(n)) ].count('aware')
                # Unaware neighbours number
                unaware = len(list(G.neighbors(n)) ) - aware

                # Oblivion factor (percent of unaware actors)
                oblivion_factor = (unaware + 0.0001) / ( (aware + 0.0001) + (unaware + 0.0001) )

                # random factor
                random_factor = np.random.uniform(0, 1)

                # probability that actor will forget information, and will not be able to pass it down
                oblivion_prob = oblivion_factor * random_factor

                # Attempt to oblivion
                if np.random.uniform(0, 1) < oblivion_prob:
                    G.nodes[n]['state'] = 'unaware'
                    
                    # increasing of engagement after oblivion
                    G.nodes[n]['engagement'] = np.round(min(1, G.nodes[n]['engagement'] * engagement_enforcement), 6)

        
        #========#
        # Kernel #
        #========#
        # If node is still aware, it disseminate information

        if G.nodes[n]['state'] == 'aware':
            
            global neighbour
            for neighbour in G.neighbors(n):
            
                if G.nodes[neighbour]['state'] == 'unaware':
                
                    #================#
                    # Weights kernel #
                    #================#
                    
                    if kernel == 'weights':
                        prob_of_internalization =  G[n][neighbour]['weight']
                    
                    #=============#
                    # WERE kernel #
                    #=============#
                    # Weights-extraversion-receptiveness-engagement
                    # kernel
                    
                    if kernel == 'WERE':
                    
                        # calculate prob_of_internalization
                        prob_of_internalization =  G[n][neighbour]['weight'] \
                        * G.nodes[neighbour]['receptiveness'] \
                        * G.nodes[neighbour]['engagement'] \
                        * G.nodes[n]['extraversion'] \
                        * WERE_multiplier
                    
                    
                    #===============#
                    # Custom kernel #
                    #===============#
    
                    if kernel == 'custom':      
                        prob_of_internalization = custom_kernel(n, neighbour)
                
                    #============================#
                    # Attempt to internalization #
                    #============================#
                    
                    if np.random.uniform(0, 1) < prob_of_internalization:
                        G.nodes[neighbour]['state'] = 'aware'
            
                #===================#
                # Engagement rising #
                #===================#
                # if node is aware, his engagement in information
                # topic may rise with given probability
                else:
                    G.nodes[neighbour]['engagement'] = \
                    np.round(G.nodes[neighbour]['engagement'] * \
                             engagement_enforcement, 6)
                        # reinforcing already informed actors

    
    #=======================#
    # Show nodes attributes #
    #=======================#
    
    # Show nodes attributes
    if show_attr == True:
        for (u, v) in G.nodes.data():
            print(u, v)     
    
    #============#
    # Draw graph #
    #============#
    
    if draw == True:
        fig_01, ax_01 = plt.subplots() # enable to plot one by one
                                       # in separate windows
        dp.draw_graph(G, pos)


    return G



#=============================================================================#
# Run n simulation steps #
#========================#

def simulation(G,  # graph object
               pos = None,  # positions of nodes
               n = 5, # number of simulation steps
               
               # wrapped args for simulation_step function
               kernel = 'weights', # simulation kernel
               custom_kernel = None, # custom simulation kernel
               WERE_multiplier = 10, # multiplier for WERE kernel
               oblivion = False, # enable information oblivion
               engagement_enforcement = 1.01,
               draw = False, # draw graph
               show_attr = False): # show attributes
    
    """ Perform n simulation steps of information diffusion for 
        a given graph.
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
        
    n : integer
        number of simulation steps for a given graph.
        
    
    Parameters wrapped from simulation_step function:
    -------------------------------------------------
        
    pos : dictionary with 2 element ndarrays as values
       Object contains positions of nodes in the graph chart. Pos is used 
       to draw the graph after simulation step.
    
    kernel : string
        Levels: "weights", "WERE", "custom"
        
        * weights - means that probability of information propagation is 
            equals to bond value between actors
        * WERE - probability of information propagation equals 
            Weights-extraversion-receptiveness-engagement equation
        * custom - probability of information propagation is computed 
            with custom function
        
    engagement_enforcement : float
        Reinforcement of agent engagement by multiplier. 
        If engagement_enforcement == 1, no reinforcement occurs.
        
        1) Agent reinforce its engagement after oblivion, 
            (later its easier to internalize information again for this
            agent, at least with WERE kernel)
        2) Agent A reinforce its engagement, during information diffusion step,
            when agent A is already informed.

    custom_kernel : function
        Function which compute probability of information propagation
        for each node in simulation step.
    
    WERE_multiplier : Float, optional
        Multiplier used for scaling WERE kernel outcome.
    
    oblivion : bool, optional
        Option which enable agents information oblivion. 
        
    draw : bool, optional
        Draw graph.

                            
    Returns
    -------
    G : graph
        A modified networkx graph object is returned after simulation.
        
    graph_list : list of lists of dictionaries
        List with statistics about simulation diffusion process.
        
        Each element of primary lists is a list. Everyinner list contains
        information about certain step of simulation, consists of information
        about nodes.
    
    avg_aware_inc_per_step: list
        Average increment of aware agents per one step of simulation.
    
    """        
    
    #=======================================#
    # append nodes data from 0 step to list #
    #=======================================#
    
    graph_list = []
    graph_list.append(copy.deepcopy(list(G.nodes.data() ) )  )
    

    #===================#
    # Run n simulations #
    #===================#
    
    for i in range(n):
        dp.simulation_step(G = G, 
                           pos = pos, 
                           
                           kernel = kernel,
                           custom_kernel = custom_kernel,
                           WERE_multiplier = WERE_multiplier, 
                           oblivion = oblivion, 
                           engagement_enforcement = engagement_enforcement,
                           draw = draw, 
                           show_attr = show_attr)

        # save nodes data to to list
        graph_list.append(copy.deepcopy(list(G.nodes.data() ) )   )
        
    
    #======================================================#
    # Count aware agents before and after simulation steps #
    #======================================================#
    
    # Check number of aware agents in 0 step
    #global aware_first
    aware_first = []
    for i in range(len(graph_list[0])):
        aware_first.append(graph_list[0][i][1]['state'])
        aware_first_c = aware_first.count('aware')
       
       # graph_list[0][1][1]['state']
        
    # Check number of aware agents in the last step
    #global aware_last
    aware_last = []
    graph_list_len = len(graph_list) - 1
    for i in range(len(graph_list[0])):
        aware_last.append(graph_list[graph_list_len][i][1]['state']) # n is the last sim
        aware_last_c = aware_last.count('aware')
            
        #graph_list[5][0][1]['state']
    
    #=================================#
    # diffusion performance measuring #
    #=================================#
    
    # equation for diffusion performance measuring
    avg_aware_inc_per_step = (aware_last_c - aware_first_c) / n
           
    # show graph statistics
    return graph_list, avg_aware_inc_per_step




#=============================================================================#
# Function for simulation sequence # 
#==================================#

def simulation_sequence(G,  # networkX graph object
                        n = 5, # number of steps in simulation
                        sequence_len = 100, # sequence of simulations
                              
                        kernel = 'weights', # kernel type
                        custom_kernel = None, # custom kernel function
                        WERE_multiplier = 10, 
                        oblivion = False, # information oblivion feature 
                        engagement_enforcement = 1.01,
                        draw = False, # draw graph
                        show_attr = False): # show nodes attributes
    
    """ Perform n simulation steps of information diffusion for 
        a given graph.
    
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
        
    sequence_len : integer
        A number of simulations to perform in one sequence.
        
        
        
    Parameters wrapped from simulation function:
    --------------------------------------------
        
    n : integer
        number of simulation steps for a given graph.
        
    
    
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
        
    WERE_multiplier : Float, optional
        Multiplier used for scaling WERE kernel outcome.
    
    oblivion : bool, optional
        Option which enable agents information oblivion. 
        
    engagement_enforcement : float
        Reinforcement of agent engagement by multiplier. 
        If engagement_enforcement == 1, no reinforcement occurs.
        
        1) Agent reinforce its engagement after oblivion, 
            (later its easier to internalize information again for this
            agent, at least with WERE kernel)
        2) Agent A reinforce its engagement, during information diffusion step,
            when another agent B is trying to pass information towards 
            agent A, but agent A is already aware.
        
    
    Returns
    -------
    
    avg_aware_inc: float
        Average increment of aware agents per simulation step for a sequence
        of simulations.
    
    
    """ 
    
    # list for storing average increment of aware agents per step
    avg_inc = []
    
    # Need to pass this arg for bug fixing
    pos = None
    # simulation f. needs this arg even if its set default as none
    
    # Run sequence of simulations
    for i in range(sequence_len):
        G_zero = copy.deepcopy(G) # Create copy of Graph for simulation i
        graph_list, avg_aware_inc_per_step \
        = dp.simulation(G_zero,  # networkX graph object
                        pos, # position of nodes
                        n, # number of steps in simulation
                              
                        kernel, # kernel type
                        custom_kernel, # custom kernel function
                        WERE_multiplier, 
                        oblivion, # information oblivion feature
                        engagement_enforcement,
                        draw, # draw graph
                        show_attr) # show nodes attributes
        
        # Append average aware agents increment per step for simulation i
        avg_inc.append(avg_aware_inc_per_step)
    
    # compute average aware agents increment per step for simulation sequence
    avg_aware_inc = sum(avg_inc) / len(avg_inc)

    return avg_aware_inc
