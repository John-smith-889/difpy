import unittest
import initialize
import networkx as nx

class TestInitialize(unittest.TestCase):
    """
    Class for testing initialize module.
    
    Class include methods for testing:
        * output objects' types
        * range of generated weights
        * range of generated nodes' attributes
        
    """
    
    #==========================#
    # Create objects for tests #
    #==========================#
    
    # This method prepare objects for all particular tests
    def setUp(self):
        print('')
        print('setUp')
    # run graph_init function
        self.G, self.pos = initialize.graph_init(n = 20, 
                                                 k= 5, 
                                                 rewire_prob = 0.1, 
                                                 initiation_perc = 0.1,
                                                 show_attr = False, 
                                                 draw_graph = True)
        
    #============================#
    # Check output objects types #
    #============================#
    
    def test_graph_init_objects_types(self):
        print('test_graph_init_objects_types')

        print(" -> Check if 'G' has proper object type")
        # Check if G is networkx graph object
        logic_value = isinstance(self.G, (nx.classes.graph.Graph))
        self.assertEqual(logic_value, True)
        
        print(" -> Check if 'pos' has proper object type")
        print('')
        # Check if pos is dictionary
        logic_value = type(self.pos) == dict
        self.assertEqual(logic_value, True)
        
    
    #================================#
    # Check output objects structure #
    #================================#
    
    # Check if output objects structure equals desired objects structures 
    def test_graph_init_nodes_quantity(self):
        print('test_graph_init_nodes_quantity')

        print(" -> Check quantity of nodes")
        print('')
        # Check if graph has 20 nodes
        nodes_number = len(self.G.nodes.data() ) == 20
        self.assertEqual(nodes_number, True)


    def test_graph_init_weights_range(self):
        print('test_graph_init_weights_range')

        # Check if weights are in (0,1] range
        # Extract weights variable
        weights = []
        for u,v,wt in self.G.edges.data('weight'):
            weights.append(wt)
            
        print(" -> Check weights min")
        weights_min = min(weights) > 0
        self.assertEqual(weights_min, True)
        
        print(" -> Check weights min")
        print('')
        weights_max = max(weights) <= 1
        self.assertEqual(weights_max, True)


    #================================#
    # Check nodes' attributes ranges #
    #================================#

    def test_graph_init_receptiveness_range(self):
        print('test_graph_init_receptiveness_range')

        # Check if receptiveness are in (0,1] range
        # Extract receptiveness variable
        receptiveness = []
        for v,wt in self.G.nodes.data('receptiveness'):
            receptiveness.append(wt)
            
        print(" -> Check receptiveness min")
        receptiveness_min = min(receptiveness) > 0
        self.assertEqual(receptiveness_min, True)

        print(" -> Check receptiveness max")
        print('')
        receptiveness_max = max(receptiveness) <= 1
        self.assertEqual(receptiveness_max, True)


    def test_graph_init_engagement_range(self):
        print('test_graph_init_engagement_range')

        # Check if engagement is in (0,1] range
        # Extract engagement variable
        engagement = []
        for v,wt in self.G.nodes.data('engagement'):
            engagement.append(wt)
            
        print(" -> Check engagement min")
        engagement_min = min(engagement) > 0
        self.assertEqual(engagement_min, True)

        print(" -> Check engagement max")
        print('')
        engagement_max = max(engagement) <= 1
        self.assertEqual(engagement_max, True)

    def test_graph_init_extraversion_range(self):
        print('test_graph_init_extraversion_range')

        # Check if extraversion is in (0,1] range
        # Extract extraversion variable
        extraversion = []
        for v,wt in self.G.nodes.data('extraversion'):
            extraversion.append(wt)
            
        print(" -> Check extraversion min")
        extraversion_min = min(extraversion) > 0
        self.assertEqual(extraversion_min, True)

        print(" -> Check extraversion max")
        print('')
        extraversion_max = max(extraversion) <= 1
        self.assertEqual(extraversion_max, True)


    def test_graph_init_state(self):
        print('test_graph_init_state')

        # check if all elements of state are all in ['aware', 'unaware']
        # Extract state variable
        state = []
        for v,wt in self.G.nodes.data('state'):
            state.append(wt)
            
        print(" -> Check state levels")
        state = len([i for i in state if i in ['unaware', 'aware']]) == len(state)
        self.assertEqual(state, True)

    #============================#
    # Check -||- something - cdn #
    #============================#



# With this line we may run tests in cmd/anaconda prompt 
# as "python test_initialize.py"
if __name__ == '__main__':
    unittest.main()
    
    
    
