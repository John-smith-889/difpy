import unittest
import initialize
import networkx as nx

class TestInitialize(unittest.TestCase):

    # This method prepare objects for all particular tests
    def setUp(self):
        print('setUp')
    # run graph_init function
        self.G, self.pos = initialize.graph_init(n = 20, 
                                                 k= 5, 
                                                 rewire_prob = 0.1, 
                                                 initiation_perc = 0.1,
                                                 show_attr = False, 
                                                 draw_graph = True)
        
    # Check if output equals desired objects 
    def test_graph_init_objects_type(self):
        
        # Check if G is networkx graph object
        logic_value = isinstance(self.G, (nx.classes.graph.Graph))
        self.assertEqual(logic_value, True)
        
        # Check if pos is networkx graph object
        logic_value = type(self.pos) == dict
        self.assertEqual(logic_value, True)
        

    # Check if output objects structure equals desired objects structures 
    def test_graph_init_objects_structure(self):
        pass    




# With this line we may run tests in cmd/anaconda prompt 
# as "python test_initialize.py"
if __name__ == '__main__':
    unittest.main()
    
    
    
