import numpy as np
import networkx as nx
'''The NumPy array is interpreted as an adjacency matrix for the graph'''
A = np.array([ [0,1,1,1,1,0], 
               [1,0,0,1,0,0], 
               [1,0,0,1,1,1], 
               [1,1,1,0,0,1], 
               [1,0,1,0,0,0], 
               [0,0,1,1,0,0] ])
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
nx.draw(G, with_labels=True)
btw = nx.betweenness_centrality(G, normalized=True)
