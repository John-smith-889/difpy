"""
Created on Wed Nov  6 05:47:09 2019

    Module enables graph initialization for simulations in difpy 
    package.

    There are possibilities to create sample difpy graph from scratch, 
    or adjust existing NetworkX. Module consists also function for 
    examine basic graph's properties.
    
    Objects
    ----------
    graph_init() : function
       A function creating graph ready for simulation purposes in difpy.

    draw_colored_graph() : function
       A function for draw colored graph.

    graph_adjust() : function
       A function adjusting networkx graph for simulation purposes in difpy.
    
    graph_stats() : function
       A function returning basic statistics of a graph and a chart.
      
"""

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#==========================================#
# Function for create graph and initialize #
#==========================================#

def graph_init(n = 26, # number of nodes
               k= 5, # number of single node neighbours before rewriting 
                     # edges
               rewire_prob = 0.1, # probability of node rewrite 
               initiation_perc = 0.1, # percent of randomly informed nodes
               show_attr = True, # show node weights and attributes
               draw = True): # probability of rewrite edge in random place
    """ Graph initialization with watts_strogatz_graph() function. 

    Create a difpy graph with added nodes attributes: relation wages, 
    awareness of an information, ...
    The graph is ready to perform simulation.
    
    
    Parameters
    ----------

    n : integer
       Nodes number of the graph.

    k : integer
        number of single node neighbours before rewriting edges
    
    rewire_prob : float
        probability of rewrite edge in random place

    legend : bool, optional
       Add legend to the graph which describes colored nodes.
    
    show_attr : bool, optional
        Show list of wages and other generated attributes of nodes.
    
    draw : bool, optional
        Draw graph.


    Returns
    -------
    G : graph
        A networkx graph object.

    pos : dictionary with 2 element ndarrays as values
       Object contains positions of nodes in the graph chart. Pos is used 
       to make drawnings of the graph during simulation. 
       
    """    
    
    #==============#
    # Create graph #
    #==============#
    
    # Global G variable
    global G
    # Create global variable pos (later used also in simulation function)
    global pos
    # Create basic watts-strogatz graph
    G = nx.watts_strogatz_graph(n = n, k = k, p = rewire_prob, seed=None)
    # Compute a position of graph elements
    pos = nx.spring_layout(G)


    #======================#
    # Add weights to graph #
    #======================#
    # Weights - are probabilities of contact between nodes of given social 
    # network.
    # Weights are randomly sampled from exponential distribution.
    
    # Values have to be scaled so we cannot add it directly to graph,
    # but after generation and scaling, and filling zeros with 0.000001
    # for computation purposes
    
    # Create ndarray of weights
    weights = np.round(np.random.exponential(scale = 0.1, 
        size = G.number_of_edges()), 6).reshape(G.number_of_edges(),1)
    
    # Scale weights to [0,1] range
    scaler = MinMaxScaler()
    scaler.fit(weights)
    scaled_weights = scaler.transform(weights)
    scaled_weights = np.round(scaled_weights, 6)
    # eliminate zeros for computation purposes
    for (x,y), i in np.ndenumerate(scaled_weights):
        if i == 0:
            scaled_weights[x,y] =0.000001
    
    # Add weights to the graph
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['weight'] = scaled_weights[i,0]

    #============================#
    # Set node attribute - state #
    #============================#
    
    # "State" Variable levels:
    # * Unaware - is actor who did not internalized the information and 
    #   is not able to pass it down. Initially, all nodes are 
    #   in state: Unaware.
    # * Aware - is the actor who internalized the information and is able 
    #   to pass it down.
    
    nx.set_node_attributes(G, 'unaware', 'state') # (G, value, key)


    #====================================#
    # Set node attribute - receptiveness #
    #====================================#
    
    # Receptiveness - general parameter of each node, expressing how much 
    # in general the actor is receptive in context of given social network.
    # Receptiveness is randomly sampled from normal distribution.

    # Create ndarray of receptiveness
    receptiveness = np.round(np.random.normal(
        size = G.number_of_edges()), 6).reshape(G.number_of_edges(),1)
    
    # Scale weights to [0,1] range
    scaler = MinMaxScaler()
    scaler.fit(receptiveness)
    scaled_receptiveness = scaler.transform(receptiveness)
    scaled_receptiveness = np.round(scaled_receptiveness, 6)
    # eliminate zeros for computation purposes
    for (x,y), i in np.ndenumerate(scaled_receptiveness):
        if i == 0:
            scaled_receptiveness[x,y] =0.000001

    # Add receptiveness parameter to nodes 
    for v in G.nodes():
        G.nodes[v]['receptiveness'] = scaled_receptiveness[v,0]

    #===================================#
    # Set node attribute - extraversion #
    #===================================#

    # Extraversion is agent eagerness to express itself to other agents
    # Extraversion is randomly sampled from normal distribution.

    # Create ndarray of extraversion
    extraversion = np.round(np.random.normal(
        size = G.number_of_edges()), 6).reshape(G.number_of_edges(),1)
    
    # Scale weights to [0,1] range
    scaler = MinMaxScaler()
    scaler.fit(extraversion)
    scaled_extraversion = scaler.transform(extraversion)
    scaled_extraversion = np.round(scaled_extraversion, 6)
    # eliminate zeros for computation purposes
    for (x,y), i in np.ndenumerate(scaled_extraversion):
        if i == 0:
            scaled_extraversion[x,y] =0.000001

    # Add receptiveness parameter to nodes 
    for v in G.nodes():
        G.nodes[v]['extraversion'] = scaled_extraversion[v,0]

    #=================================#
    # Set node attribute - engagement #
    #=================================#

    # Engagement - engagement with the information related topic, 
    # strengthness of the experiences connected with information topic.
    # How much the information is objectivly relevant for actor. 
    # Engagement is randomly sampled from exponential distribution.

    # Create ndarray of engagement
    engagement = np.round(np.random.exponential(
        size = G.number_of_edges()), 6).reshape(G.number_of_edges(),1)
    
    # Scale weights to [0,1] range
    scaler = MinMaxScaler()
    scaler.fit(engagement)
    scaled_engagement = scaler.transform(engagement)
    scaled_engagement = np.round(scaled_engagement, 6)
    # eliminate zeros for computation purposes
    for (x,y), i in np.ndenumerate(scaled_engagement):
        if i == 0:
            scaled_engagement[x,y] =0.000001

    # Add receptiveness parameter to nodes 
    for v in G.nodes():
        G.nodes[v]['engagement'] = scaled_engagement[v,0]
    
    #===================#
    # Random initiation #
    #===================#

    # Compute number of nodes
    N = G.number_of_nodes()
    # Return list of numbers of randomly aware agents
    infected_agents_id = random.sample(population = range(0,N), 
                                       k = int(N * initiation_perc))
    # Set those nodes as aware
    for v in infected_agents_id:
        G.nodes[v]['state'] = 'aware'
    
    # Show nodes attributes
    if show_attr == True:
        print("Node attributes:")
        for (u, v) in G.nodes.data():
            print(u, v)     
    
        # Check how scaled weights looks like
        x = list(range(len(scaled_weights)))
        scaled_weights = np.sort(scaled_weights, axis = 0)
        # show numbered values
        dict_0 = dict(zip(x,scaled_weights))
        print("Wages:")
        for u, v in dict_0.items():
            print(u, v) 

    # Draw graph
    if draw == True:    
        draw_graph(G = G, pos = pos)
    # draw_colored_graph_2
    return G, pos


#================================#
# Function for drawing the graph #
#================================#

def draw_graph(G, # graph
               pos, # position of nodes
               aware_color = '#f63f89',
               not_aware_color = '#58f258',
               legend = True):
    
    """ Draw the graph G using Matplotlib and NetworkX.

    Draw the graph with Matplotlib and NetworkX with two colors associated 
    with 2 types of agents - aware of certain information, and unaware one.
    Legend describing nodes is optional.
    
    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary with 2 element ndarrays as values
       Positions of nodes

    aware_color : string
       Specify the color of nodes aware of certain information.

    not_aware_color : string
       Specify the color of nodes unaware of certain information.

    legend : bool, optional
       Add legend to the graph which describes colored nodes.
       
    """
    
    # Create variables for store nodes numbers    
    color_map_1 = []   
    color_map_2 = []
    # Create list of nodes numbers which are 'aware'
    awarelist = [i for i, d in G.nodes.data() if d['state'] == 'aware' ]
    # Create list of nodes numbers which are not 'aware'
    notawarelist = [i for i in range(len(G.nodes.data())) if i not in awarelist]
    # Append strings about colors to color_map lists
    for node in G:
        if node in awarelist:
            color_map_1.append(aware_color) # aware
        else: color_map_2.append(not_aware_color) # not aware
    # Draw the graph
    plt.title("Graph")
    nx.draw_networkx_nodes(G,pos = pos, nodelist = awarelist, 
                           node_color = color_map_1, with_labels = True, 
                           label='Aware agent', alpha = 0.7)
    nx.draw_networkx_nodes(G,pos = pos, nodelist = notawarelist, 
                           node_color = color_map_2, with_labels = True, 
                           label='Not aware agent', alpha = 0.7)
    nx.draw_networkx_labels(G, pos = pos, font_size=12, font_color='k', 
                            font_family='sans-serif', font_weight='normal', 
                            alpha=1.0)
    nx.draw_networkx_edges(G,pos=pos)
    
    # optional legend
    if legend == True:
        plt.legend(numpoints = 1)
    
