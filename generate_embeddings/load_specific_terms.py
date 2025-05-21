import pickle
import numpy as np # Though not strictly needed for this minimal example, often useful with embeddings

TERM_EMBEDDINGS_PATH = "data/embeddings/term_embeddings.pkl"
RESOURCE_EMBEDDINGS_PATH = "data/embeddings/resource_embeddings.pkl"

def load_pickle_file(file_path):
    """Loads a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
    except Exception as e:
        print(f"ERROR: Could not load {file_path}: {e}")
    return None

def find_term_embedding(data, term_to_find):
    """
    Finds the embedding for a specific term in term_embeddings.pkl format.
    Assumes data is a list of dicts, each with 'term' and 'embedding' (list of floats).
    """
    if not isinstance(data, list):
        print("Term embeddings data is not a list.")
        return None
        
    for entry in data:
        if isinstance(entry, dict) and entry.get('term') == term_to_find:
            embedding_vector = entry.get('embedding')
            if isinstance(embedding_vector, list):
                return embedding_vector
            else:
                print(f"Found term '{term_to_find}', but its embedding is not a list: {type(embedding_vector)}")
                return None
    print(f"Term '{term_to_find}' not found in term embeddings.")
    return None

def find_resource_embedding(data, term_to_find):
    """
    Finds the embedding for a specific term in resource_embeddings.pkl format.
    Assumes data is a list of dicts, each with 'term' and 'embedding'.
    'embedding' is a list containing an object with a '.values' attribute (list of floats).
    """
    if not isinstance(data, list):
        print("Resource embeddings data is not a list.")
        return None

    for entry in data:
        if isinstance(entry, dict) and entry.get('term') == term_to_find:
            embedding_container = entry.get('embedding')
            if isinstance(embedding_container, list) and len(embedding_container) > 0:
                embedding_object = embedding_container[0]
                if hasattr(embedding_object, 'values'):
                    embedding_vector = getattr(embedding_object, 'values')
                    if isinstance(embedding_vector, list):
                        return embedding_vector
                    else:
                        print(f"Found term '{term_to_find}', but its .values attribute is not a list: {type(embedding_vector)}")
                        return None
                else:
                    print(f"Found term '{term_to_find}', but its embedding object lacks a .values attribute.")
                    return None
            else:
                print(f"Found term '{term_to_find}', but its 'embedding' field is not a list or is empty: {type(embedding_container)}")
                return None
    print(f"Term '{term_to_find}' not found in resource embeddings.")
    return None

def print_embedding_summary(term_name, embedding_vector):
    """Prints a summary of the found embedding."""
    if embedding_vector:
        print(f"Embedding for '{term_name}':")
        print(f"  Type: {type(embedding_vector)}")
        print(f"  Length: {len(embedding_vector)}")
        print(f"  First 5 elements: {embedding_vector[:5]}...")
    else:
        print(f"Embedding not found or invalid for '{term_name}'.")
    print("-" * 30)

if __name__ == "__main__":
    # --- Load Term Embeddings ---
    print("Loading Term Embeddings...")
    term_data = load_pickle_file(TERM_EMBEDDINGS_PATH)
    
    if term_data:
        term1_name = "17th century dance" # From your example
        term1_embedding = find_term_embedding(term_data, term1_name)
        print_embedding_summary(term1_name, term1_embedding)

    # --- Load Resource Embeddings ---
    print("\nLoading Resource Embeddings...")
    resource_data = load_pickle_file(RESOURCE_EMBEDDINGS_PATH)

    if resource_data:
        term2_name = "abortion" # From your example
        term2_embedding = find_resource_embedding(resource_data, term2_name)
        print_embedding_summary(term2_name, term2_embedding)

    print("\nScript finished.") 