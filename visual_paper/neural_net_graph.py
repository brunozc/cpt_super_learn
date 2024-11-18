import matplotlib.pyplot as plt
import networkx as nx

# Define the structure of the neural network
layers = [6, 7, 7, 5]  # Input layer (7), two hidden layers (10 each), output layer (2)

# Create the graph
G = nx.DiGraph()

# Add nodes and edges
positions = {}
node_counter = 0

for layer_idx, num_nodes in enumerate(layers):
    for node_idx in range(num_nodes):
        node_name = f"L{layer_idx}N{node_idx}"
        G.add_node(node_name, layer=layer_idx)
        positions[node_name] = (layer_idx, -node_idx)
        if layer_idx > 0:  # Connect to previous layer
            for prev_node_idx in range(layers[layer_idx - 1]):
                prev_node_name = f"L{layer_idx-1}N{prev_node_idx}"
                G.add_edge(prev_node_name, node_name)

# Draw the network
plt.figure(figsize=(10, 6))
nx.draw(
    G,
    pos=positions,
    with_labels=False,
    node_size=500,
    node_color="blue",
    edge_color="gray",
    alpha=0.7
)
plt.title("Neural Network Visualization", fontsize=16)
plt.axis("off")
plt.show()
