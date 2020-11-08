import networkx as nx
import matplotlib.pyplot as plt


#Plotting existing graphs
G = nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
plt.savefig("karate_graph.pdf")
plt.clf()



#Generating random graphs
from scipy.stats import bernoulli

bernoulli.rvs(p=0.2)
N = 20
p = 0.2

def er_graph(N, p):
    """Generaate an ER graph"""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes:
        for node2 in G.nodes:  #need inner loop to  potentially add an edge between node 1 and node2
            if node1 < node2 and bernoulli.rvs(p=p): #need node1<node2 otherwise each node will be considered twice by the for loops#1 is interpreted as True, 0 as False, using == True is not needed
                G.add_edge(node1, node2)
    return G

nx.draw(er_graph(50,0.08), node_size = 40, node_color="gray")
plt.savefig("er1.pdf")
plt.clf()


#plotting degree distribution

def plot_degree_distribution(G):
    a = [d for n, d in G.degree()]  #returns a Degreeview object, but we want a list
    plt.hist(a, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.legend(['G1', 'G2'])
    plt.title("Degree Distribution")

G1 = er_graph(100, 0.03)
G2 = er_graph(100, 0.30)
#G3 = er_graph(500, 0.08)

plot_degree_distribution(G1)
plot_degree_distribution(G2)
#plot_degree_distribution(G3)
plt.savefig("er_degree_dist.pdf")
plt.clf()

import numpy as np

#import adjacency matricies
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=",")
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=",")

#convert adjacency matricies to graph objects
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

#calculating the degree of each node, the degree
def basic_net_stats(G):
    a = G.number_of_nodes()
    b = G.number_of_edges()
    degree_seq = [d for n,d in G.degree()]
    print("Number of Nodes: %d" %a)
    print("Number of edges: %d" %b)
    print("Average degree: %.2f" % np.mean(degree_seq))
plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig("village degree dists.pdf")
plt.clf()


#create generator function
def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)
        
gen = connected_component_subgraphs(G1)
#g = gen.__next__() #this jumps to the next component, remember a component is a group of connected nodes. returns the  size (num nodes) of the component
#the above is slow and tedious. to find the max we can simply;
G1_LCC = max(connected_component_subgraphs(G1), key=len)#finds the largest connected component. Need to tell it what max means, in this case, it is the length, the number of nodes
G2_LCC = max(connected_component_subgraphs(G2), key=len) 

plt.figure()
nx.draw(G1_LCC, node_color="red", edge_color="gray", node_size=20)
plt.savefig("Village1_connectivity.pdf")
plt.clf()

plt.figure()
nx.draw(G2_LCC, node_color="green", edge_color="gray", node_size=20)
plt.savefig("Village2_connectivity.pdf")
plt.clf()