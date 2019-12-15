"""
Created on Thu Dec 12 01:02:53 2019

    Module enables simulation in Difpy package.

    Objects
    ----------
    simulation_step() : function
       A function creating graph ready for simulation purposes in difpy.

"""

import difpy as dp
# import networkx as nx # used only by difpy subfunction
import numpy as np
# import random # used only by difpy subfunction
import matplotlib.pyplot as plt


#==============================#
# Function one simulation step # ver 1
#==============================#

def simulation_step(G, # NetworkX graph
                    pos = None,
                    kernel = 'weights',
                    custom_kernel = None,
                    WERE_multiplier = 10, 
                    oblivion = False, 
                    engagement_enforcement = 1.01,
                    draw = False, 
                    show_attr = False):

    """ Perform one simulation step of information diffusion.
    
    
    Parameters
    ----------

    G : graph
        A networkx graph object.
        
        To use default information propagation kernels, its nodes need 
        to have weights, and extraversion, receptiveness, engagement 
        parameters.

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
        
    engagement_enforcement : Float
        Enforcement of agent engagement by multiplier. 
        If engagement_enforcement == 1, no enforcement occurs.
        
        1) Agent enforce its engagement after oblivion, 
            (later its easier to internalize information again for this
            agent, at least with WERE kernel)
        2) Agent A enforce its engagement, during information diffusion step,
            when another agent B is trying to pass information towards 
            agent A, but agent A is already aware.
        
    
    draw : bool, optional
        Draw graph.


    Returns
    -------
    G : graph
        A networkx graph object.

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
                        # enforcing already informed actors

    
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




