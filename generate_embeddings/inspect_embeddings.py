import pickle
import numpy as np

TERM_EMBEDDINGS_PATH = "data/embeddings/term_embeddings.pkl"
RESOURCE_EMBEDDINGS_PATH = "data/embeddings/resource_embeddings.pkl"

def summarize_vector_data(vector_data, indent="    "):
    """Prints a summary of a list or NumPy array assumed to be an embedding vector."""
    if not isinstance(vector_data, (list, np.ndarray)):
        print(f"{indent}Data is not a list or NumPy array (Type: {type(vector_data)}). Cannot summarize as vector.")
        print(f"{indent}Value (partial): {str(vector_data)[:100]}...")
        return

    print(f"{indent}Vector Type: {type(vector_data)}")
    list_representation = None
    if isinstance(vector_data, np.ndarray):
        print(f"{indent}Vector Shape: {vector_data.shape}")
        try:
            list_representation = vector_data.tolist() if vector_data.ndim > 0 else [vector_data.item()]
        except AttributeError: 
            list_representation = list(vector_data) # Fallback
    elif isinstance(vector_data, list):
        print(f"{indent}Vector Length: {len(vector_data)}")
        list_representation = vector_data
    else: # Should not be reached due to the initial type check
        print(f"{indent}Unexpected vector data type: {type(vector_data)}")
        print(f"{indent}Value (partial): {str(vector_data)[:100]}...")
        return

    if isinstance(list_representation, list):
        # Ensure all elements in the sample are numeric for clean printing
        sample_elements = []
        for x in list_representation[:5]:
            if isinstance(x, (int, float, np.number)):
                sample_elements.append(x)
            else:
                sample_elements.append(str(type(x))) # Show type if not numeric
        
        suffix = "..." if len(list_representation) > 5 else ""
        print(f"{indent}First few elements: {sample_elements}{suffix}" if list_representation else f"{indent}Elements: []")
    else:
        print(f"{indent}Could not represent as simple list for sampling. Raw data sample: {str(vector_data)[:100]}...")

def print_sample_entry_custom(entry):
    if isinstance(entry, dict):
        print("Sample entry (dictionary with custom embedding display):")
        for key, value in entry.items():
            is_common_embedding_key = key in ['embedding', 'embedding_vector_np']

            if is_common_embedding_key:
                print(f"  '{key}': (Attempting to summarize as embedding)")
                actual_vector = None
                detection_info = "Unknown structure"

                if isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'values'):
                    potential_vector = getattr(value[0], 'values')
                    if isinstance(potential_vector, (list, np.ndarray)):
                        actual_vector = potential_vector
                        detection_info = "list containing object with .values (e.g., [ContentEmbedding])"
                elif hasattr(value, 'values') and actual_vector is None:
                    potential_vector = getattr(value, 'values')
                    if isinstance(potential_vector, (list, np.ndarray)):
                        actual_vector = potential_vector
                        detection_info = "object with .values (e.g., ContentEmbedding)"
                elif isinstance(value, (list, np.ndarray)) and actual_vector is None:
                    actual_vector = value
                    detection_info = "direct list or NumPy array"
                
                print(f"    Detected structure for '{key}': {detection_info}")
                if actual_vector is not None:
                    summarize_vector_data(actual_vector, indent="    ")
                else:
                    print(f"    Could not auto-detect a vector in '{key}'. Original type: {type(value)}")
                    print(f"    Value (partial): {str(value)[:200]}...")
            elif key == 'values' and isinstance(value, (list, np.ndarray)):
                 print(f"  '{key}': (Interpreted as direct embedding vector)")
                 summarize_vector_data(value, indent="    ")
            else:
                print(f"  '{key}': {value}")
    else:
        print(f"Sample entry (not a dict, type: {type(entry)}): {str(entry)[:500]}...")

def load_and_sample_pickle(file_path, description):
    print(f"--- Attempting to load: {description} ---")
    print(f"File path: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not data and data != 0 and data != False: # Check if data is truly empty or None
            print("File loaded, but it appears to be empty or contains no data (e.g. None, empty list/dict).")
            print("-" * 40 + "\n")
            return

        sample_to_inspect = None
        data_type_str = str(type(data))

        if isinstance(data, list):
            data_type_str = "list"
            print(f"Successfully loaded. Type: {data_type_str}, Total entries: {len(data)}")
            if len(data) > 0:
                sample_to_inspect = data[0]
            else:
                print("Data is an empty list.")
        elif isinstance(data, dict):
            data_type_str = "dict"
            print(f"Successfully loaded. Type: {data_type_str}, Total keys: {len(data)}")
            if len(data) > 0:
                first_key = next(iter(data))
                print_sample_entry_custom({first_key: data[first_key]})
                print("-" * 40 + "\n")
                return # Handled by print_sample_entry_custom for dicts
            else:
                print("Data is an empty dictionary.")
        else:
            print(f"Successfully loaded. Data type: {data_type_str}")
            # For non-list/dict data, print a small part directly
            print(f"Data (partial): {str(data)[:500]}...")

        if sample_to_inspect is not None: # This path is mainly for when data was a non-empty list
            if isinstance(sample_to_inspect, dict):
                print_sample_entry_custom(sample_to_inspect)
            elif isinstance(sample_to_inspect, (list, np.ndarray)):
                print("Sample entry (detected as direct list/array, summarizing as vector):")
                summarize_vector_data(sample_to_inspect, indent="  ")
            else: # Fallback for unknown sample structure from a list
                print(f"Sample entry (from list, type {type(sample_to_inspect)}):")
                print(str(sample_to_inspect)[:500] + ("..." if len(str(sample_to_inspect)) > 500 else ""))
        elif isinstance(data, list) and len(data) == 0: # Handled above, but defensive
            pass # Message already printed
        elif not isinstance(data, dict): # If it wasn't a list or dict initially
             pass # Message and sample already printed
            
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
    except pickle.UnpicklingError:
        print(f"ERROR: Could not unpickle the file at {file_path}. It might be corrupted or not a pickle file.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading {file_path}: {e}")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    load_and_sample_pickle(TERM_EMBEDDINGS_PATH, "Term Embeddings")
    load_and_sample_pickle(RESOURCE_EMBEDDINGS_PATH, "Resource Embeddings") 