import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def visualize_graph(G):
    # draw the graph
    nx.draw(G, with_labels = True, font_size = 8, node_size = 5, node_color = 'skyblue', edge_color = 'gray')
    plt.show()

def create_and_visualize_graph(file_path):
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
    pos = nx.spring_layout(G, seed = 42, k = 1)
    node_sizes = [15 + 10 * G.degree(n) for n in G.nodes()]
    plt.figure(figsize=(16, 12), facecolor='white')
    nx.draw_networkx_nodes(G, pos, 
                            node_size=node_sizes, 
                            node_color='lightblue', 
                            alpha=0.7, 
                            edgecolors='steelblue')
    nx.draw_networkx_edges(G, pos, 
                            width=1, 
                            edge_color='gray', 
                            alpha=0.2, 
                            arrows=True, 
                            connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, 
                             font_size=7, 
                             font_color='darkblue', 
                             font_weight='normal')
    
    plt.title('Country Name Connection Graph', fontsize=15, fontweight='light')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    create_and_visualize_graph('datasets/countries_and_cities.csv')