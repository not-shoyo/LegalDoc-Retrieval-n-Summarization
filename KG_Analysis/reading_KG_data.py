import pandas as pd

def parse_entity(entity_str):
  # Remove leading and trailing spaces
  entity_str = entity_str.strip()

  # Check if the entity string is a node or a relationship
  if entity_str.startswith("(:") and entity_str.endswith(")"):
    # It's a node, parse it
    return parse_node(entity_str)
  elif entity_str.startswith("[:") and entity_str.endswith("]"):
    # It's a relationship, parse it
    return parse_relationship(entity_str)
  else:
    raise ValueError("Invalid entity format: {}".format(entity_str))

def parse_node(node_str):
  # Extract the node content
  node_start = 2
  node_end = -1
  content = node_str[node_start: node_end]

  if node_str.find('{') != -1:
    node_type, content = map(str.strip, content.split('{', 1))
    # Parse the content using a custom function
    attributes = parse_attributes(content[:-1])
    return {'type': node_type, 'attributes': attributes}
  else: 
    return {'type': node_str, 'attributes': None}


def parse_relationship(relationship_str):
  # Extract relationship type and attributes
  relation_start = 2
  relation_end = -1

  if relationship_str[relation_start: relation_end].find('{') != -1:
    relationship_type, attributes_str = map(str.strip, relationship_str[relation_start:relation_end].split('{', 1))
    # Parse the attributes using a custom function
    attributes = parse_attributes(attributes_str[:-1])
    return {'type': relationship_type, 'attributes': attributes}
  else: 
    return {'type': relationship_str[relation_start: relation_end], 'attributes': None}

def parse_attributes(attributes_str):
  # Custom function to parse attributes without quotes
  attributes = {}
  prev_key = ''
  for pair in attributes_str.split(','):
    try:
      key, value = map(str.strip, pair.split(':', 1))
      attributes[key] = value
      prev_key = key
    except:
      attributes[prev_key] = attributes[prev_key] + pair
  return attributes

def read_knowledge_graph(csv_file):
  # Read CSV file into a pandas DataFrame
  df = pd.read_csv(csv_file, sep=',')

  # Create dictionaries to store entities and relations
  entities = {}
  relations = {}

  # Process each row in the DataFrame
  for index, row in df.iterrows():
    source_entity = parse_entity(row['n'])
    destination_entity = parse_entity(row['m'])
    relation = parse_entity(row['r'])

    # Store information in entities and relations dictionaries
    entities[index] = source_entity
    entities[index] = destination_entity
    relations[index] = relation

  return entities, relations

if __name__ == "__main__":
    query_csv_file_path = 'All_Data/Generated_Data/KG_Data/Sample_KG_Data/Sample_KG_Query_AILA_Q1.csv'
    query_entities, query_relations = read_knowledge_graph(query_csv_file_path)

    document_csv_file_path = 'All_Data/Generated_Data/KG_Data/Sample_KG_Data/Sample_KG_Document_AILA_C9.csv'
    document_entities, document_relations = read_knowledge_graph(document_csv_file_path)

    # # Extract entity types from both query and document entities
    # query_entity_types = {info['type'] for info in query_entities.values()}
    # document_entity_types = {info['type'] for info in document_entities.values()}

    # # Find the common entity types
    # common_entity_types = query_entity_types.intersection(document_entity_types)

    # # Print the set of common entity types
    # print("Common Entity Types:")
    # print(common_entity_types)

    # # Extract relation types from both query and document relations
    # query_relation_types = {info['type'] for info in query_relations.values()}
    # document_relation_types = {info['type'] for info in document_relations.values()}

    # # Find the common relation types
    # common_relation_types = query_relation_types.intersection(document_relation_types)

    # # Print the set of common relation types
    # print("Common Relation Types:")
    # print(common_relation_types)
