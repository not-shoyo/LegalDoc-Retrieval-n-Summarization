# Create Siamese GNN Architecture

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Embedding, Layer, Concatenate, Dense

from graph_representation import query_triplets, document_triplets, query_adjacency_matrix, document_adjacency_matrix, query_node_features, document_node_features, relevance_label

query_edge_list, document_edge_list = query_triplets, document_triplets

"""
  # Define Node and Edge Embedding Layers
  node_embedding_layer = ...
  edge_embedding_layer = ...
"""

def define_embedding_layers(num_nodes, node_embedding_dim, num_edge_types, edge_embedding_dim):
    """
    Define node and edge embedding layers.
    
    Args:
    - num_nodes: Number of nodes in the graph.
    - node_embedding_dim: Dimensionality of node embeddings.
    - num_edge_types: Number of edge types in the graph.
    - edge_embedding_dim: Dimensionality of edge embeddings.
    
    Returns:
    - node_embedding_layer: Node embedding layer.
    - edge_embedding_layer: Edge embedding layer.
    """
    # Define node embedding layer
    node_embedding_layer = Embedding(input_dim=num_nodes, output_dim=node_embedding_dim, name='node_embedding')

    # Define edge embedding layer
    edge_embedding_layer = Embedding(input_dim=num_edge_types, output_dim=edge_embedding_dim, name='edge_embedding')
    
    return node_embedding_layer, edge_embedding_layer

# Example adjacency matrix and edge list
# query_adjacency_matrix = np.array([[0, 1, 1],
#                                     [1, 0, 0],
#                                     [1, 0, 0]])

# query_edge_list = [{'source': 'Node1', 'destination': 'Node2', 'relation': 'Relation1'},
#                    {'source': 'Node1', 'destination': 'Node3', 'relation': 'Relation2'},
#                    {'source': 'Node2', 'destination': 'Node1', 'relation': 'Relation3'}]

# Determine the number of nodes and edge types
num_nodes = query_adjacency_matrix.shape[0]
num_edge_types = len(set(edge[1]["type"] for edge in query_edge_list))

# Define embedding layers
node_embedding_dim = 64
edge_embedding_dim = 32
node_embedding_layer, edge_embedding_layer = define_embedding_layers(num_nodes, node_embedding_dim, num_edge_types, edge_embedding_dim)

# Test embedding layers
node_ids = [0, 1, 2]
node_embeddings = node_embedding_layer(np.array(node_ids))
print("Node Embeddings:")
print(node_embeddings)

# edge_types = [0, 1, 2]
# edge_embeddings = edge_embedding_layer(np.array(edge_types))
# print("\nEdge Embeddings:")
# print(edge_embeddings)

"""
  # Graph Convolutional Layers
  graph_convolution_layers = ...
"""

class GraphConvolutionLayer(Layer):
    def __init__(self, num_filters, activation='relu', **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], self.num_filters),
                                      initializer='glorot_uniform',
                                      trainable=True)
    
    def call(self, inputs):
        features, adjacency_matrix = inputs
        print(f"features.shape: ${features.shape}")
        print(f"adjacency_matrix.shape: ${adjacency_matrix.shape}")
        output = tf.matmul(adjacency_matrix, features)  # Graph convolution
        output = tf.matmul(output, self.kernel)  # Apply weights
        output = self.activation(output)  # Apply activation function
        return output

# Example usage:
num_filters = 64
graph_convolution_layer = GraphConvolutionLayer(num_filters)

# Assuming 'query_node_features' and 'document_node_features' are the node features obtained previously
# 'query_adjacency_matrix' and 'document_adjacency_matrix' are the adjacency matrices

query_features = tf.constant(list(query_node_features.values()))
document_features = tf.constant(list(document_node_features.values()))
query_adjacency_matrix = tf.constant(query_adjacency_matrix)
document_adjacency_matrix = tf.constant(document_adjacency_matrix)

# Reshape features to match expected input shape
query_features = tf.expand_dims(query_features, axis=0)  # Adding batch dimension
document_features = tf.expand_dims(document_features, axis=0)  # Adding batch dimension

query_output = graph_convolution_layer((query_features, query_adjacency_matrix))
document_output = graph_convolution_layer((document_features, document_adjacency_matrix))

print("Query Graph Convolution Output Shape:", query_output.shape)
print("Document Graph Convolution Output Shape:", document_output.shape)

"""
  # Pooling Layers
  pooling_layer = ...
"""

class GlobalPoolingLayer(Layer):
    def __init__(self, pooling_type='mean', **kwargs):
        super(GlobalPoolingLayer, self).__init__(**kwargs)
        self.pooling_type = pooling_type

    def call(self, inputs):
        if self.pooling_type == 'mean':
            return tf.reduce_mean(inputs, axis=0)  # Compute mean pooling
        elif self.pooling_type == 'max':
            return tf.reduce_max(inputs, axis=0)   # Compute max pooling
        elif self.pooling_type == 'sum':
            return tf.reduce_sum(inputs, axis=0)   # Compute sum pooling
        else:
            raise ValueError("Invalid pooling type. Choose from 'mean', 'max', or 'sum'.")

# Example usage:
pooling_type = 'mean'  # You can change this to 'max' or 'sum' if desired
global_pooling_layer = GlobalPoolingLayer(pooling_type)

# Assuming 'query_output' and 'document_output' are the output tensors from the graph convolutional layers
query_graph_representation = global_pooling_layer(query_output)
document_graph_representation = global_pooling_layer(document_output)

print("Query Graph Representation Shape:", query_graph_representation.shape)
print("Document Graph Representation Shape:", document_graph_representation.shape)

# Model Parameters
query_input = [query_adjacency_matrix] # [query_adjacency_matrix, query_edge_list]
document_input = [document_adjacency_matrix] # [document_adjacency_matrix, document_edge_list]
labels = relevance_label
batch_size = 16 #(16 to 256)
num_epochs = 10 #(10 to 50+)
validation_split = 0.2

# Siamese Architecture
query_branch = Sequential([node_embedding_layer, graph_convolution_layer, global_pooling_layer]) # [node_embedding_layer, edge_embedding_layer, graph_convolution_layer, global_pooling_layer]
document_branch = Sequential([node_embedding_layer, graph_convolution_layer, global_pooling_layer]) # [node_embedding_layer, edge_embedding_layer, graph_convolution_layer, global_pooling_layer]

# Call Siamese Branches
query_branch_output = query_branch(*query_input)
document_branch_output = document_branch(*document_input)

# Merge Layers
merge_layer = Concatenate()([query_branch_output, document_branch_output])

# Similarity Calculation
similarity_output = Dense(1, activation='sigmoid')(merge_layer)

# Define Model
siamese_gnn_model = Model(inputs=[query_branch_output, document_branch_output], outputs=similarity_output)

# Compile Model
siamese_gnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
siamese_gnn_model.fit(x=[query_input, document_input], y=labels, batch_size=batch_size, epochs=num_epochs, validation_split=validation_split)
