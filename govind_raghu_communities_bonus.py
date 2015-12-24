import sys
import networkx as nx
import community
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import colors

def buildgraph():
    # Open input file containing graph edges
    inputfilename = sys.argv[1]
    inputfile = open(inputfilename, 'r')
    # Initialize networkx graph
    G = nx.Graph()
    # Add all edges into the graph
    for line in inputfile:
        line = line.rstrip('\n')
        nodes = map(int, line.split())
        G.add_edge(*nodes)
    # Return the constructed graph
    return G

def BFS(G, s):
    # node_distance stores distance of every node from source
    node_distance = {}
    # patch_counts stores total number of path counts from source to node
    path_counts = dict.fromkeys(G, 0.0)
    # preds stores list of predecesors for every node
    preds = defaultdict(list)
    # Stores intermediate list of visited nodes of BFS
    visited = []
    # Initialize node distance and path_counts for source node
    node_distance[s] = 0
    path_counts[s] = 1.0
    # Push source node to queue
    Q = [s]
    # Iterate till no more nodes to visit
    while Q:
        # Pop node from Queue
        v = Q.pop(0)
        # Append node to visited list
        visited.append(v)
        # Check out all adjacent nodes to current node
        for adj in G[v]:
            # Check if node is aldready visited
            if adj not in node_distance:
                # push new node to Queue
                Q.append(adj)
                # Update node distance of new node
                node_distance[adj] = node_distance[v] + 1
            # If this is the shortest distance to this node
            if node_distance[adj] == node_distance[v] + 1:
                # Update path counts based on the predecessor
                path_counts[adj] += path_counts[v]
                # Add v to predecessor list of adj node
                preds[adj].append(v)
    # Return list of all visited nodes, predessesors of each node and pathcounts
    return visited, preds, path_counts

def update_betweeness(betweenness, visited, preds, path_counts):
    # Initialize credits of all nodes to 0
    credits = dict.fromkeys(visited, 0)
    # Go through all nodes in visited list
    while visited:
        # Take a node from visited list
        u = visited.pop()
        # Calculate credits to be given to each predecesors
        creditfraction = (1.0 + credits[u]) / path_counts[u]
        # For every predecessor of node u
        for v in preds[u]:
            # Calculate new credit to be updated on edges
            c = path_counts[v] * creditfraction
            # Update betweeness values of edge u,v
            if (v, u) not in betweenness:
                betweenness[(u, v)] += c
            else:
                betweenness[(v, u)] += c
            # Update credits of the predecessor node
            credits[v] += c
    # Resutn betweeness values of edges
    return betweenness

def calc_edge_betweeness(G):
    # Initialize betweeneness odf all edges to 0
    betweenness = dict.fromkeys(G.edges(), 0.0)
    # For every node in graph G
    for node in G:
        # Do a BFS on the node
        visited, preds, path_counts = BFS(G, node)
        # Update edge betweeness values
        betweenness = update_betweeness(betweenness, visited, preds, path_counts)
    # Divide edge betweeness values by 2
    for edge in betweenness:
        betweenness[edge] *= 0.5
    # Return edge betweeness values
    return betweenness

def girvan_newman_iter(G):
    # Calculate Edge Betweeness using Girvan Newman algorithm
    edge_betweeness = calc_edge_betweeness(G)
    # Get the max value of edge betweeness
    max_betweeness = max(edge_betweeness.values())
    # Get all edges corresponding to max betweeness value
    max_edges = [k for k,v in edge_betweeness.items() if v==max_betweeness]
    # Delete all max betweeness edges
    G.remove_edges_from(max_edges)

def getpartitions(G):
    # Get all connected components of graph
    components = nx.connected_components(G)
    # Convert to partition format that is accepted by modularity function
    partitions = {}
    for i, component in enumerate(components):
        parition = {node:i for node in component}
        partitions.update(parition)
    # Return partition dictionary
    return partitions

def getcomponents(partition):
    # Convert partition format to components containing list of nodes
    components = defaultdict(list)
    componentlist = []
    for k,v in partition.items():
       components[v].append(k)
    for component in components:
        componentlist.append(sorted(components[component]))
    return componentlist

if __name__ == '__main__':
    G = buildgraph()
    orig = G.copy()
    numedges = len(G.edges())
    # Get partitions of initial graph
    bestpartition = getpartitions(G)
    # Calculate modularity of partitions
    bestmodularity = community.modularity(bestpartition, G)
    # Do Girvan Newman edge removal till no mode edges remain
    while(numedges > 0):
        # Perform Girvan Newman algorithm to remove edge with highest betweeness
        girvan_newman_iter(G)
        # Get partitions of graph after edge removal
        partitions = getpartitions(G)
        # Update number of edges
        numedges = len(G.edges())
        # If edges still exist in graph
        if(numedges > 0):
            # Calculate modularity of new partition
            modularity = community.modularity(partitions, orig)
            # Update best partition if new partition is better
            if(modularity > bestmodularity):
                bestmodularity = modularity
                bestpartition = partitions

    # Draw the graph
    pos = nx.spring_layout(orig)
    plt.axis('off')
    componentlist = getcomponents(bestpartition)

    # Draw nodes
    nodecolors = colors.cnames.values()
    n = len(nodecolors)
    i = 0
    for component in componentlist:
        print component
        nx.draw_networkx_nodes(orig, pos, nodelist=component, node_size=350, node_color=nodecolors[i])
        i = (i+4)%n

    # Draw all edges
    nx.draw_networkx_edges(orig, pos, edgelist=orig.edges(), width=1)
    # Draw labels on nodes
    nx.draw_networkx_labels(orig, pos, font_size=10, font_family='sans-serif')

    # Save graph to output image
    outputfile = sys.argv[2]
    plt.savefig(outputfile)
    # plt.show()
