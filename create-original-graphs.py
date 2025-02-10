'''
This code creates and visualizes the graph using the networkx library in python.
The code reads the csv file and creates a graph with each country as a node.
If the last character of the source country is the same as the first character of the destination country, 
then a directed edge from the source country to the destination copuntry is added.
'''

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def create_and_visualize_graph(file_path, output_file):
    data = pd.read_csv(file_path)
    # create a directed graph
    G = nx.DiGraph()
    # add nodes and edges to the graph
    for i, row in data.iterrows():
        G.add_node(row['Name'])
    for i, source_row in data.iterrows():
        for j, destination_row in data.iterrows():
            if i != j and source_row['Name'][-1].lower() == destination_row['Name'][0].lower():
                G.add_edge(source_row['Name'], destination_row['Name'])
    # visualize the graph
    pos = nx.spring_layout(G, k = 3, iterations = 200, scale = 10)
    
    # Calculate node sizes with degree centrality
    degree_centrality = nx.degree_centrality(G)
    node_sizes = [1000 * degree_centrality[node] + 5 for node in G.nodes()]

    plt.figure(figsize = (20, 15), facecolor = 'white')
    
    # Draw edges with red color scheme and curves
    edges = nx.draw_networkx_edges(G, pos, 
        edge_color = '#cc9988', 
        alpha = 0.2, 
        width = 0.5, 
        arrows = True, 
        arrowsize = 10, 
        connectionstyle = 'arc3, rad = 0.2')
    
    # Draw nodes with red color scheme -- the darker the color, the higher the degree centrality
    nodes = nx.draw_networkx_nodes(G, pos, 
        node_size = node_sizes, 
        node_color = [degree_centrality[node] for node in G.nodes()], 
        cmap = plt.cm.Reds, 
        alpha = 0.7, 
        edgecolors = 'none')
    
    # Draw labels
    labels = nx.draw_networkx_labels(G, pos, 
        font_size = 6, 
        font_color = 'black', 
        font_weight = 'light')
    
    plt.title('Country Name Connection Graph', fontsize = 15, fontweight = 'light')
    plt.axis('off')
    plt.tight_layout()
    
    # Add more whitespace around the plot
    plt.margins(x = 0.2, y = 0.2)
    plt.savefig(output_file)

if __name__ == '__main__':
    create_and_visualize_graph('datasets/countries_only.csv', 'original_graphs/countries_graph.png')
    create_and_visualize_graph('datasets/cities_only.csv', 'original_graphs/cities_graph.png')
    create_and_visualize_graph('datasets/countries_and_cities.csv', 'original_graphs/countries_and_cities_graph.png')