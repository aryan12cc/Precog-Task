'''
This code creates and visualizes the graph using the networkx library in python.
The code reads the csv file and creates a graph with each country as a node.
The graph created by this code is a directed weighted graph between each letter of the English alphabet.
The edges are created if we can go from one node to another by saying a city / country.
For example: By saying Afghanistan, one can go from A to N, so a directed edge from A to N is created.
To avoid multiple edges, each directed edge has a "weight" attached to it, denoting the number of options
one can say to go from one node to another.
'''

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# a function to create and visualise the graph
def create_and_visualize_graph(file_path, output_file):
    data = pd.read_csv(file_path)
    # create a directed graph
    G = nx.DiGraph()
    # add nodes and edges to the graph
    for char in 'abcdefghijklmnopqrstuvwxyz':
        G.add_node(char.upper())
    for i, source_row in data.iterrows():
        if G.has_edge(source_row['Name'][0].upper(), source_row['Name'][-1].upper()):
            G[source_row['Name'][0].upper()][source_row['Name'][-1].upper()]['weight'] += 1
        else:
            G.add_edge(source_row['Name'][0].upper(), source_row['Name'][-1].upper(), weight = 1)
    # visualize the graph
    pos = nx.spring_layout(G, k = 5, iterations = 10, scale = 15)
    
    # assign node sizes based on degree centrality
    degree_centrality = nx.degree_centrality(G)
    node_sizes = [200 + 1000 * degree_centrality[node] for node in G.nodes()]
    plt.figure(figsize = (24, 18), facecolor = 'white')
    
    # Draw edges with red color scheme and curves
    edges = nx.draw_networkx_edges(G, pos, 
        edge_color = '#cc9988', 
        alpha = 0.3, 
        width = 1.0, 
        arrows = True, 
        arrowsize = 15, 
        connectionstyle = 'arc3, rad = 0.2', 
        min_source_margin = 20, 
        min_target_margin = 20)
    
    # Draw nodes with red color scheme -- the darker the color, the higher the degree centrality
    nodes = nx.draw_networkx_nodes(G, pos, 
        node_size = node_sizes, 
        node_color = 'red', 
        alpha = 0.8, 
        edgecolors = 'white', 
        linewidths = 2)
    
    # Draw labels
    labels = nx.draw_networkx_labels(G, pos, 
        font_size = 12, 
        font_color = 'black', 
        font_family = 'sans-serif')

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, 
        edge_labels = edge_labels, 
        label_pos = 0.5,
        connectionstyle = 'arc3, rad = 0.2',
        rotate = False)
    
    plt.title('Country Name Connection Graph', 
        fontsize = 20, 
        fontweight = 'bold', 
        pad = 20, 
        fontfamily = 'sans-serif')
    plt.axis('off')
    plt.tight_layout(pad = 2.0)
    plt.margins(x = 0.3, y = 0.3)
    plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')

if __name__ == '__main__':
    create_and_visualize_graph('datasets/countries_only.csv', 'simplified_graphs/countries_graph.png')
    create_and_visualize_graph('datasets/cities_only.csv', 'simplified_graphs/cities_graph.png')
    create_and_visualize_graph('datasets/countries_and_cities.csv', 'simplified_graphs/countries_and_cities_graph.png')