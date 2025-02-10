'''
Code for community detection algorithms (task 2)
I have used the Louvain and Infomap algorithms

Louvain and Infomap algorithm both require an undirected graph
'''

import networkx as nx
import community as community_louvain
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# a function to create a graph consisting of the countries data in the file_path
def create_graph(file_path):
    data = pd.read_csv(file_path)
    G = nx.DiGraph()
    # add nodes
    for i, source_row in data.iterrows():
        G.add_node(source_row['Name'])
    # add edges
    for i, source_row in data.iterrows():
        for j, dest_row in data.iterrows():
            if source_row['Name'][-1].upper() == dest_row['Name'][0].upper():
                G.add_edge(source_row['Name'], dest_row['Name'])
    return G

# a function to visualize the partitioned communities created by the louvain algorithm
def plot_louvain(graph, partition, title):
    pos = nx.spring_layout(graph, k = 3, iterations = 200, scale = 10)

    community_map = defaultdict(list)
    country_community = defaultdict(int)
    # add to the community
    for node, comm in partition.items():
        community_map[comm].append(node)
        country_community[node] = comm

    # plot the graph
    plt.figure(figsize = (20, 15), facecolor = 'white')

    edges = nx.draw_networkx_edges(graph, pos, 
        edge_color = '#cc9988', 
        alpha = 0.2, 
        width = 0.5, 
        arrows = True, 
        arrowsize = 10, 
        connectionstyle = 'arc3, rad = 0.2')
    colors = [partition[node] for node in graph.nodes()]

    nodes = nx.draw_networkx_nodes(graph, pos, 
        node_color = colors, 
        alpha = 0.7, 
        edgecolors = 'none')

    labels = nx.draw_networkx_labels(graph, pos, 
        font_size = 6, 
        font_color = 'black', 
        font_weight = 'light')

    plt.title(title)
    plt.savefig('task2-results/louvain_communities.png')

# a function to calculate the coverage of the communities
def louvain_coverage(graph, communities):
    internal_edges = sum(1 for u, v in graph.edges() if any(u in comm and v in comm for comm in communities))
    total_edges = graph.number_of_edges()
    coverage = internal_edges / total_edges if total_edges > 0 else 0
    return coverage

def louvain_algorithm(graph):
    # get the partition and score using the louvain algorithm
    partition = community_louvain.best_partition(graph)
    modularity_score = community_louvain.modularity(partition, graph)
    
    # Convert partition dictionary to list of communities
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    
    community_list = list(communities.values())

    # make the partition a list
    partition_dict = {}
    for idx, comm in enumerate(community_list):
        for node in comm:
            if idx not in partition_dict:
                partition_dict[idx] = set()
            partition_dict[idx].add(node)
    partition_list = list(partition_dict.values())

    partition_quality = nx.community.partition_quality(graph, partition_list)

    with open('task2-results/louvain_communities_results.txt', 'w') as f:
        for idx, comm in enumerate(community_list):
            f.write(f'Community {idx + 1}: {", ".join(comm)}\n')
        f.write(f'Louvain Modularity: {modularity_score}\n')
        f.write(f'Louvain Coverage: {partition_quality[0]}\n')
        f.write(f'Louvain Conductance: {1 - partition_quality[1]}\n')

    plot_louvain(graph, partition, 'Community Detection - Louvain')

# a function to plot the communities detected by the infomap algorithm
def plot_infomap(graph, communities, title):
    pos = nx.spring_layout(graph, k = 3, iterations = 200, scale = 10)

    community_map = defaultdict(int)
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx

    plt.figure(figsize = (20, 15), facecolor = 'white')

    edges = nx.draw_networkx_edges(
        graph, pos, 
        edge_color = '#cc9988', 
        alpha = 0.2, 
        width = 0.5, 
        arrows = True, 
        arrowsize = 10, 
        connectionstyle = 'arc3, rad=0.2'
    )
    colors = [community_map.get(node, 0) for node in graph.nodes()]

    nodes = nx.draw_networkx_nodes(graph, pos, 
        node_color = colors, 
        alpha = 0.7, 
        edgecolors = 'none', 
    )

    labels = nx.draw_networkx_labels(graph, pos, 
        font_size = 6, 
        font_color = 'black', 
        font_weight = 'light')

    plt.title(title)
    plt.savefig('task2-results/infomap_communities.png')

# function to implement the infomap algorithm
def infomap_algorithm(graph):
    # get the partition and score using the infomap algorithm
    graph_ig = ig.Graph.TupleList(graph.edges(), directed = True)
    
    # igraph node indices to country names
    index_to_country = {idx: name for idx, name in enumerate(graph.nodes())}
    infomap_communities = graph_ig.community_infomap()
    
    # communities from index to country names
    country_communities = [[index_to_country[idx] for idx in comm] for comm in infomap_communities]
    infomap_modularity = graph_ig.modularity(infomap_communities)

    partition_quality = nx.community.partition_quality(graph, country_communities)
    
    # print('Infomap Communities: ', country_communities)
    with open('task2-results/infomap_communities_results.txt', 'w') as f:
        for idx, comm in enumerate(country_communities):
            f.write(f'Community {idx + 1}: {", ".join(comm)}\n')
        f.write(f'Infomap Modularity: {infomap_modularity}\n')
        f.write(f'Infomap Coverage: {partition_quality[0]}\n')
        f.write(f'Infomap Conductance: {1 - partition_quality[1]}\n')

    plot_infomap(graph, country_communities, 'Community Detection - Infomap')

def main():
    graph = create_graph('datasets/countries_only.csv')
    graph_undirected = graph.to_undirected()
    louvain_algorithm(graph_undirected)
    infomap_algorithm(graph_undirected)

if __name__ == '__main__':
    main()