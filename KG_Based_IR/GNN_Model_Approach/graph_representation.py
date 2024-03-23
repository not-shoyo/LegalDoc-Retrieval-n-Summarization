# Represent each query and document as a graph, where nodes represent entities or concepts, and edges represent relationships.
# Assign feature vectors to nodes and edges, representing attributes or properties of each entity or relationship.
# Construct adjacency matrices or edge lists to represent the connectivity of nodes and edges within each graph.

"""
  pip install transformers
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from reading_KG_data import return_required_triplets, return_relevance_label


def assign_features(triplets, tokenizer, model):
    """
    Assigns LegalBERT embeddings as feature vectors to nodes and edges based on the given triplets.
    
    Args:
    - triplets: List of dictionaries containing source, destination, and relation texts.
    - tokenizer: LegalBERT tokenizer.
    - model: LegalBERT model.
    
    Returns:
    - node_features: Dictionary mapping node IDs to LegalBERT embeddings.
    - edge_features: List of dictionaries containing edge features.
    """
    node_features = {}
    edge_features = []
    
    # Assign features for nodes
    # n -> 0
    # r -> 1
    # m -> 2
    for triplet in triplets:
        source = triplet[0]["type"]
        destination = triplet[2]["type"]
        
        if source not in node_features:
            # Tokenize and obtain LegalBERT embeddings for source node text
            source_tokens = tokenizer(source, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                source_embedding = model(**source_tokens).last_hidden_state.mean(dim=1).squeeze(0).numpy()
            node_features[source] = source_embedding
        
        if destination not in node_features:
            # Tokenize and obtain LegalBERT embeddings for destination node text
            destination_tokens = tokenizer(destination, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                destination_embedding = model(**destination_tokens).last_hidden_state.mean(dim=1).squeeze(0).numpy()
            node_features[destination] = destination_embedding
        
        # Assign features for edges
        edge_features.append({
            'source': source,
            'destination': destination,
            'relation': triplet[1]["type"]
        })
    
    return node_features, edge_features

def construct_adjacency_matrix(triplets, node_features):
    """
    Constructs an adjacency matrix based on the given triplets and node features.
    
    Args:
    - triplets: List of dictionaries containing source, destination, and relation texts.
    - node_features: Dictionary mapping node IDs to feature vectors.
    
    Returns:
    - adjacency_matrix: Numpy array representing the adjacency matrix.
    """
    node_ids = set(node_features.keys())
    num_nodes = len(node_ids)
    node_index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    
    # n -> 0
    # r -> 1
    # m -> 2
    for triplet in triplets:
        source_id = triplet[0]["type"]
        destination_id = triplet[2]["type"]
        
        if source_id in node_ids and destination_id in node_ids:
            source_idx = node_index_map[source_id]
            destination_idx = node_index_map[destination_id]
            
            # Assuming the graph is undirected, set adjacency for both directions
            adjacency_matrix[source_idx, destination_idx] = 1
            adjacency_matrix[destination_idx, source_idx] = 1
    
    return adjacency_matrix

# Load LegalBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")


# Example query and document triplets
# query_triplets = [
#     {'n': 'Node1', 'm': 'Node2', 'r': 'Relation1'},
#     {'n': 'Node2', 'm': 'Node3', 'r': 'Relation2'},
#     {'n': 'Node3', 'm': 'Node4', 'r': 'Relation3'}
# ]

# document_triplets = [
#     {'n': 'NodeA', 'm': 'NodeB', 'r': 'RelationA'},
#     {'n': 'NodeB', 'm': 'NodeC', 'r': 'RelationB'},
#     {'n': 'NodeC', 'm': 'NodeD', 'r': 'RelationC'}
# ]


query_num = 1
document_num = 9

query_triplets = return_required_triplets(f"All_Data/Generated_Data/KG_Data/Sample_KG_Data/Sample_KG_Query_AILA_Q{query_num}.csv")
document_triplets = return_required_triplets(f"All_Data/Generated_Data/KG_Data/Sample_KG_Data/Sample_KG_Document_AILA_C{document_num}.csv")
relevance_label = return_relevance_label(query_num, document_num)

# print(query_triplets)
# print(document_triplets)

# Assign features and construct adjacency matrices for query and document graphs
query_node_features, query_edge_features = assign_features(query_triplets, tokenizer, model)
document_node_features, document_edge_features = assign_features(document_triplets, tokenizer, model)

query_adjacency_matrix = construct_adjacency_matrix(query_triplets, query_node_features)
document_adjacency_matrix = construct_adjacency_matrix(document_triplets, document_node_features)

print("Query Node Features:", query_node_features)
print("Query Edge Features:", query_edge_features)
print("Query Adjacency Matrix:", query_adjacency_matrix)

print("\nDocument Node Features:", document_node_features)
print("Document Edge Features:", document_edge_features)
print("Document Adjacency Matrix:", document_adjacency_matrix)

print("\n========================================================================================================\n")