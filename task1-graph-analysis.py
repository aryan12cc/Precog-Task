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
from string import ascii_uppercase
import numpy as np
from collections import defaultdict

# function to create the simplified graph using the data in the csv file path
def create_graph(file_path):
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
    return G

# function to check which nodes are reachable from a given source node
def check_reachability(graph, start):
    vis = dict()
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        vis[char] = False
    stack = [start]
    while stack:
        node = stack.pop()
        vis[node] = True
        for neighbor in graph.neighbors(node):
            if not vis[neighbor]:
                stack.append(neighbor)
    return vis

# function to get the degree (in degree and out degree) of each node in the graph
def find_degree(graph, graph_name):
    # find the in-degree and out-degree of each node
    in_degrees = dict()
    out_degrees = dict()
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        in_degrees[char] = 0
        out_degrees[char] = 0
    # add self loops to both in-degrees and out-degrees
    for edge in graph.edges():
        out_degrees[edge[0]] += graph[edge[0]][edge[1]]['weight']
        in_degrees[edge[1]] += graph[edge[0]][edge[1]]['weight']
    # print the in-degree and out-degree of each node
    stuck_nodes = []
    node_degrees = []
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        node_degrees.append((char, in_degrees[char], out_degrees[char]))
        if in_degrees[char] > out_degrees[char]:
            stuck_nodes.append(char)
    with open (f'task1-results/degree/{graph_name}.txt', 'w') as f:
        f.write(f'Node Degrees for {graph_name}\n')
        for node in node_degrees:
            f.write(f'Node {node[0]} has in-degree {node[1]} and out-degree {node[2]}\n')
        f.write('Stuck Nodes\n')
        for node in stuck_nodes:
            f.write(f'A player can get stuck at node {node}\n')

# function to check whether a player can get trapped in any closed subgraph of <= 3 nodes
def find_closed_subgraphs(graph, graph_name):
    # remove edges that cannot be reached (eg you can't say Zambia in countries graph because you cannot reach Z)
    vis = check_reachability(graph, 'S')
    remove_edges = []
    for edge in remove_edges:
        graph.remove_edge(edge[0], edge[1])
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if not vis[char] and graph.has_node(char) == True:
            graph.remove_node(char)
    # find the in-degree and out-degree of each node
    in_degrees = dict()
    out_degrees = dict()
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        in_degrees[char] = 0
        out_degrees[char] = 0
    # add self loops to both in-degrees and out-degrees
    for edge in graph.edges():
        out_degrees[edge[0]] += graph[edge[0]][edge[1]]['weight']
        in_degrees[edge[1]] += graph[edge[0]][edge[1]]['weight']
    list_of_closed_subgraphs = []
    # check for all 3 node closed subgraphs and check if the player can be trapped in them
    for node1 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        for node2 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for node3 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                # unreachable nodes
                if in_degrees[node1] == 0 or in_degrees[node2] == 0 or in_degrees[node3] == 0:
                    continue
                cnt = dict()
                for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    cnt[char] = 0
                for edge in graph.edges():
                    # can go outside of this subgraph
                    if edge[0] == node1 or edge[0] == node2 or edge[0] == node3:
                        cnt[edge[1]] += graph[edge[0]][edge[1]]['weight']
                    # can come inside this subgraph
                    if edge[1] == node1 or edge[1] == node2 or edge[1] == node3:
                        cnt[edge[0]] -= graph[edge[0]][edge[1]]['weight']
                tot_cnt = 0
                for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    # either stays, or has more options to come in than to go out for any node
                    if cnt[char] <= 0 or char == node1 or char == node2 or char == node3:
                        tot_cnt += 1
                if tot_cnt == 26:
                    list_of_closed_subgraphs.append((node1, node2, node3))
    with open(f'task1-results/closed_subgraphs/{graph_name}.txt', 'w') as f:
        f.write(f'Closed Subgraphs for {graph_name}\n')
        for subgraph in list_of_closed_subgraphs:
            f.write(f'The closed subgraph is {subgraph}\n')

# function to check the reachability of each node from a given source node
# this is done by getting the shortest path from the source node to all other nodes, and then plotting the graph
def get_reachability(graph, graph_name):
    # matrix that stores the reachability of each node from a given source node
    reachability_matrix = np.full((26, 26), 7) # 7 is the maximum distance between any two nodes
    for source_node in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        reachability_matrix[ord(source_node) - 65][ord(source_node) - 65] = 0
        queue = [source_node]
        while queue:
            node = queue.pop(0)
            for next_char in graph.neighbors(node):
                if reachability_matrix[ord(source_node) - 65][ord(next_char) - 65] == 7:
                    reachability_matrix[ord(source_node) - 65, ord(next_char) - 65] = reachability_matrix[ord(source_node) - 65, ord(node) - 65] + 1
                    queue.append(next_char)
    # plot the reachability matrix
    fig, ax = plt.subplots()
    color_ax = ax.imshow(reachability_matrix, cmap = 'Blues', interpolation = 'nearest')
    cbar = fig.colorbar(color_ax, ax = ax, ticks = [0, 1, 2, 3, 4, 5, 6, 7])
    cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', 'Unreachable'])

    ax.set_ylabel('Source Node')
    ax.set_yticks(range(26))
    ax.set_yticklabels(list(ascii_uppercase))
    ax.set_xlabel('Destination Node')
    ax.set_xticks(range(26))
    ax.set_xticklabels(list(ascii_uppercase))
    ax.set_title('Reachability Matrix')
    plt.savefig(f'task1-results/reachability/{graph_name}.png')

# function to check whether there is a correlation between centrality and winning
# across 1'000 games, we check the average of the centrality of all the nodes chosen by the winning player and the losing player
# plot a graph on avg winning centrality - avg losing centrality
# plot the average line as well
def centrality_correlation(graph, graph_name):
    game_results = []
    centrality_differences = []
    # create the graph as a multiedge directed graph for better degree centrality calculation
    new_graph = nx.MultiDiGraph()
    for node in graph.nodes():
        new_graph.add_node(node)
    for edge in graph.edges():
        for cnt in range(graph[edge[0]][edge[1]]['weight']):
            new_graph.add_edge(edge[0], edge[1])
    for game in range(1000):
        game_graph = new_graph.copy()
        current_node = 'S'
        total_centrality_count_player1 = nx.degree_centrality(graph)[current_node]
        total_centrality_count_player2 = 0
        move_count_player1 = 1
        move_count_player2 = 0
        while True:
            weights_sum = len(list(game_graph.neighbors(current_node)))
            # no outgoing edges
            if weights_sum == 0:
                break
            # randomly pick an edge to go to
            random_edge = np.random.randint(0, weights_sum)
            for neighbor in game_graph.neighbors(current_node):
                random_edge -= 1
                if random_edge < 0:
                    # simulate the move
                    if move_count_player1 > move_count_player2:
                        total_centrality_count_player2 += nx.degree_centrality(game_graph)[neighbor]
                        move_count_player2 += 1
                    else:
                        total_centrality_count_player1 += nx.degree_centrality(game_graph)[neighbor]
                        move_count_player1 += 1
                    # remove the edge as you cannot choose it again
                    game_graph.remove_edge(current_node, neighbor)
                    # change the node
                    current_node = neighbor
                    break
        # calculate total centrality count for each player
        total_centrality_count_player1 /= move_count_player1
        total_centrality_count_player2 /= move_count_player2
        centrality_differences.append(total_centrality_count_player1 - total_centrality_count_player2)
        # player1 wins the game
        if move_count_player1 > move_count_player2:
            game_results.append(1)
        # player2 wins the game
        else:
            game_results.append(-1)
    # plot the results
    centrality_differences_lost = [-x for i, x in enumerate(centrality_differences) if game_results[i] == -1]
    centrality_differences_won = [x for i, x in enumerate(centrality_differences) if game_results[i] == 1]

    fig, axes = plt.subplots(1, 2, figsize = (10, 6), dpi = 80)

    axes[0].hist(centrality_differences_won, bins = 50, color = 'blue', edgecolor = 'black')
    axes[0].set_xlabel('Winning Centrality - Losing Centrality')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Centrality Difference When Player 1 Wins')

    axes[1].hist(centrality_differences_lost, bins = 50, color = 'blue', edgecolor = 'black')
    axes[1].set_xlabel('Winning Centrality - Losing Centrality')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Centrality Difference When Player 2 Wins')

    print(f'correlation for {graph_name}: {np.corrcoef(game_results, centrality_differences)[0, 1]}')
    plt.tight_layout()
    plt.savefig(f'task1-results/centrality_correlation/{graph_name}.png')

# function that simulates 10'000 games and finds the nodes where the players get stuck at the end
# helps visualize the nodes where the players are more likely to get stuck
def find_endings(graph, graph_name):
    endings_list = dict()
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        endings_list[char] = 0
    for game in range(10000):
        game_graph = graph.copy()
        current_node = 'S'
        while True:
            weights_sum = len(list(game_graph.neighbors(current_node)))
            # no outgoing edges
            if weights_sum == 0:
                endings_list[current_node] += 1
                break
            # randomly pick an edge to go to
            random_edge = np.random.randint(0, weights_sum)
            for neighbor in game_graph.neighbors(current_node):
                random_edge -= 1
                if random_edge < 0:
                    # remove the edge as you cannot choose it again
                    game_graph.remove_edge(current_node, neighbor)
                    # change the node
                    current_node = neighbor
                    break
    # plot the results
    plt.figure(figsize = (10, 6), dpi = 80)
    plt.bar(endings_list.keys(), endings_list.values(), color = 'blue')
    plt.xlabel('Node')
    plt.ylabel('Frequency')
    plt.title('Frequency of Endings on a Node')
    plt.savefig(f'task1-results/frequency_endings/{graph_name}.png')

# function to find the k-core values of the graph
# k-core is the maximal subgraph in which every vertex has atleast degree K
# reference: https://www.baeldung.com/cs/graph-k-core
def k_core_values(graph, graph_name):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    core_values = nx.core_number(graph)
    with open(f'task1-results/k_core_values/{graph_name}.txt', 'w') as f:
        f.write(f'K-Core Values for {graph_name}\n')
        for node in core_values:
            f.write(f'Node {node} has k-core value {core_values[node]}\n')

if __name__ == '__main__':
    # create the graphs of the countries
    countries_graph = create_graph('datasets/countries_only.csv')
    cities_graph = create_graph('datasets/cities_only.csv')
    countries_and_cities_graph = create_graph('datasets/countries_and_cities.csv')
    print('Properties are listed below:\n 1. Degree of each node\n 2. Closed subgraphs\n 3. Reachability Matrix')
    print(' 4. Centrality Correlation with Winning\n 5. Frequency of Endings on a node\n 6. K-core decomposition of the graph')
    property_number = int(input('Enter which property do you want to analyse (1-6)\n'))
    # show the properties of the graph based on the property selected
    if property_number == 1:
        find_degree(countries_graph, 'countries')
        find_degree(cities_graph, 'cities')
        find_degree(countries_and_cities_graph, 'countries_and_cities')
    elif property_number == 2:
        find_closed_subgraphs(countries_graph, 'countries')
        find_closed_subgraphs(cities_graph, 'cities')
        find_closed_subgraphs(countries_and_cities_graph, 'countries_and_cities')
    elif property_number == 3:
        get_reachability(countries_graph, 'countries')
        get_reachability(cities_graph, 'cities')
        get_reachability(countries_and_cities_graph, 'countries_and_cities')
    elif property_number == 4:
        centrality_correlation(countries_graph, 'countries')
        centrality_correlation(cities_graph, 'cities')
        centrality_correlation(countries_and_cities_graph, 'countries_and_cities')
    elif property_number == 5:
        find_endings(countries_graph, 'countries')
        find_endings(cities_graph, 'cities')
        find_endings(countries_and_cities_graph, 'countries_and_cities')
    elif property_number == 6:
        k_core_values(countries_graph, 'countries')
        k_core_values(cities_graph, 'cities')
        k_core_values(countries_and_cities_graph, 'countries_and_cities')