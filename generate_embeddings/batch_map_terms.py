import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Explicitly set GOOGLE_API_KEY in current process environment if it exists in .env
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    print(f"GOOGLE_API_KEY found and set in environment", file=sys.stderr)
else:
    print(f"WARNING: GOOGLE_API_KEY not found in environment variables", file=sys.stderr)

def find_jsonl_files(archive_dir):
    """Finds all .jsonl files in the given directory."""
    jsonl_files = []
    if not os.path.isdir(archive_dir):
        print(f"Error: Archive directory '{archive_dir}' not found.")
        return jsonl_files

    for root, _, files in os.walk(archive_dir):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def convert_jsonl_to_txt_names_only(jsonl_filepath, txt_filepath_out):
    """
    Reads a .jsonl file, extracts 'name' fields, and writes them to a .txt file, one per line.
    Returns True if successful and at least one name was written, False otherwise.
    """
    names_written = 0
    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as infile, \
             open(txt_filepath_out, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line)
                    name = data.get("name")
                    if name is not None and isinstance(name, str):
                        outfile.write(name + '\n')
                        names_written += 1
                    elif name is None:
                        print(f"  Info: 'name' field not found in {jsonl_filepath} line {line_num}. Skipping for .txt conversion.")
                    else: # Not a string
                        print(f"  Info: 'name' field in {jsonl_filepath} line {line_num} is not a string ('{name}'). Skipping for .txt conversion.")
                except json.JSONDecodeError:
                    print(f"  Warning: Could not decode JSON in {jsonl_filepath} line {line_num}. Skipping for .txt conversion.")
        if names_written > 0:
            print(f"Successfully converted {jsonl_filepath} to {txt_filepath_out}, extracted {names_written} names.")
            return True
        else:
            print(f"No names extracted from {jsonl_filepath} into {txt_filepath_out}.")
            if os.path.exists(txt_filepath_out):
                try:
                    if os.path.getsize(txt_filepath_out) == 0:
                        os.remove(txt_filepath_out)
                        print(f"Removed empty output file: {txt_filepath_out}")
                except OSError as e:
                    print(f"Error removing potentially empty file {txt_filepath_out}: {e}")
            return False
    except Exception as e:
        print(f"Error during conversion of {jsonl_filepath} to {txt_filepath_out}: {e}")
        if os.path.exists(txt_filepath_out):
            try:
                os.remove(txt_filepath_out)
            except OSError:
                pass 
        return False

def run_mapping_script(script_path, input_names_list):
    """Runs the map_input_to_term.py script with a list of input names and captures its output, expecting a JSON list."""
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        target_script_path = os.path.join(current_script_dir, script_path)

        if not os.path.exists(target_script_path):
            target_script_path = script_path # Fallback

        python_executable = sys.executable
        command = [python_executable, target_script_path] + input_names_list
        
        # Create a copy of the current environment to pass to the subprocess
        env = os.environ.copy()
        
        # Ensure GOOGLE_API_KEY is in the environment
        if os.getenv("GOOGLE_API_KEY"):
            env["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        print(f"  Running mapping script with {len(input_names_list)} terms in a batch...", file=sys.stderr)
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env  # Pass the environment variables
        )
        
        if process.stderr:
            # map_input_to_term.py might print extensive logs to stderr for the whole batch.
            # Consider how much of this to display or log.
            print(f"  Stderr from '{target_script_path}' (batch run):\n{process.stderr.strip()}", file=sys.stderr)

        if process.returncode != 0:
            return {
                "status": "script_error", 
                "error_message": f"Error running '{target_script_path}' for batch. Exit code: {process.returncode}",
                "details": process.stdout.strip(), # Include stdout for clues
                "is_batch_error": True # Flag to indicate this error is for the whole batch
            }
        
        try:
            # Expecting a JSON list from map_input_to_term.py
            return json.loads(process.stdout.strip())
        except json.JSONDecodeError as e:
            return {
                "status": "json_decode_error",
                "error_message": f"Failed to decode JSON list output from '{target_script_path}' for batch: {e}",
                "raw_output": process.stdout.strip(),
                "is_batch_error": True
            }

    except FileNotFoundError:
        print(f"Error: The script '{script_path}' was not found. Make sure it's in the correct path.", file=sys.stderr)
        return {"status": "script_not_found", "error_message": f"Script '{script_path}' not found.", "is_batch_error": True}
    except Exception as e:
        print(f"An unexpected error occurred while running '{script_path}' for batch: {e}", file=sys.stderr)
        return {"status": "execution_exception", "error_message": str(e), "is_batch_error": True}

def main():
    parser = argparse.ArgumentParser(description="Batch process .jsonl files: extract 'name' fields, map all names in a single batch using map_input_to_term.py, and save results to a JSON file.")
    parser.add_argument(
        "--archive_dir",
        type=str,
        default="archive",
        help="Directory containing .jsonl files (default: 'archive')."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="archive_mapping_results.json",
        help="File to save all mapping results as a single JSON list (default: 'archive_mapping_results.json')."
    )
    parser.add_argument(
        "--map_script_path",
        type=str,
        default="map_input_to_term.py",
        help="Path to the map_input_to_term.py script (default: 'map_input_to_term.py')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of terms to process in a single call to the mapping script (default: 1000)."
    )
    args = parser.parse_args()

    print("Starting batch processing with intermediate .txt file creation...")
    print(f"Archive directory: {args.archive_dir}")
    print(f"Output file (JSON): {args.output_file}")
    print(f"Mapping script: {args.map_script_path}")

    jsonl_files = find_jsonl_files(args.archive_dir)
    if not jsonl_files:
        print("No .jsonl files found in archive directory. Exiting.")
        return

    print(f"Found {len(jsonl_files)} .jsonl files to process.")

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for .txt name files: {temp_dir}")
    
    files_to_process_info = []
    for jsonl_file_path in jsonl_files:
        base_name = os.path.basename(jsonl_file_path)
        # Sanitize base_name if it could contain problematic characters for a filename, though os.path.basename should be safe.
        txt_file_name = base_name + ".names.txt"
        temp_txt_filepath = os.path.join(temp_dir, txt_file_name)
        
        print(f"Converting {jsonl_file_path} to {temp_txt_filepath}...")
        if convert_jsonl_to_txt_names_only(jsonl_file_path, temp_txt_filepath):
            files_to_process_info.append({
                "original_jsonl": jsonl_file_path,
                "names_txt": temp_txt_filepath
            })
        else:
            print(f"Skipping {jsonl_file_path} due to conversion failure or no names extracted.")

    if not files_to_process_info:
        print("No .jsonl files were successfully converted to .txt name lists. Exiting.")
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
        return

    print(f"Successfully converted {len(files_to_process_info)} .jsonl files to .txt name lists in {temp_dir}.")
    
    all_mapping_results = []
    processed_records_count = 0
    source_name_pairs = [] # Store (original_jsonl_path, name_to_map)

    try:
        print("Collecting all names for batch processing...")
        for file_info in files_to_process_info:
            original_jsonl_path = file_info["original_jsonl"]
            names_txt_path = file_info["names_txt"]
            
            try:
                with open(names_txt_path, 'r', encoding='utf-8') as name_file:
                    for line_number, name_to_map_raw in enumerate(name_file, 1):
                        name_to_map = name_to_map_raw.strip()
                        if not name_to_map: 
                            print(f"  Skipping empty line {line_number} in {names_txt_path}.", file=sys.stderr)
                            continue
                        source_name_pairs.append((original_jsonl_path, name_to_map))
            except Exception as e:
                print(f"Error reading names from {names_txt_path} for batch collection: {e}", file=sys.stderr)
        
        if not source_name_pairs:
            print("No names collected for batch processing. Exiting.")
            # shutil.rmtree(temp_dir) # Cleanup is in finally block
            return

        print(f"Collected {len(source_name_pairs)} names. Will process in batches of {args.batch_size}.")
        
        total_source_pairs = len(source_name_pairs)
        for i in range(0, total_source_pairs, args.batch_size):
            chunk_source_name_pairs = source_name_pairs[i:i + args.batch_size]
            chunk_names_to_map_only = [pair[1] for pair in chunk_source_name_pairs]
            
            if not chunk_names_to_map_only:
                continue # Should not happen if source_name_pairs is not empty

            print(f"Processing batch {i // args.batch_size + 1}/{(total_source_pairs + args.batch_size - 1) // args.batch_size} (terms {i+1}-{min(i + args.batch_size, total_source_pairs)} of {total_source_pairs})...")
            batch_mapping_output = run_mapping_script(args.map_script_path, chunk_names_to_map_only)

            if isinstance(batch_mapping_output, dict) and batch_mapping_output.get("is_batch_error"):
                print(f"Critical error in batch {i // args.batch_size + 1}: {batch_mapping_output.get('error_message', 'Unknown batch error')}", file=sys.stderr)
                if batch_mapping_output.get('raw_output'):
                    print(f"Raw output from script: {batch_mapping_output['raw_output']}", file=sys.stderr)
                elif batch_mapping_output.get('details'):
                    print(f"Details from script: {batch_mapping_output['details']}", file=sys.stderr)
                # Decide if you want to stop all processing or skip this batch and continue
                print(f"Skipping further processing for this batch due to critical error.", file=sys.stderr)
                continue # Continue to the next batch
            
            elif isinstance(batch_mapping_output, list):
                if len(batch_mapping_output) == len(chunk_source_name_pairs):
                    for j, mapping_output_obj in enumerate(batch_mapping_output):
                        original_jsonl_path, input_name = chunk_source_name_pairs[j]
                        
                        if "input_term" in mapping_output_obj and mapping_output_obj["input_term"] != input_name:
                            print(f"Warning: Mismatch in order/content for batch {i // args.batch_size + 1}, item {j}. Expected '{input_name}', got '{mapping_output_obj['input_term']}'. Associating anyway.", file=sys.stderr)

                        final_mapped_term = None
                        if isinstance(mapping_output_obj, dict):
                            final_mapped_term = mapping_output_obj.get("mapped_term")

                        result_entry = {
                            "source_jsonl_file": original_jsonl_path,
                            "input_name": input_name,
                            "final_canonical_term": final_mapped_term,
                            "mapping_output": mapping_output_obj
                        }
                        all_mapping_results.append(result_entry)
                        processed_records_count += 1
                else:
                    print(f"Error in batch {i // args.batch_size + 1}: Number of results ({len(batch_mapping_output)}) does not match number of inputs ({len(chunk_source_name_pairs)}). Skipping this batch.", file=sys.stderr)
            else:
                print(f"Error in batch {i // args.batch_size + 1}: Unexpected output type from mapping script. Expected a list, got {type(batch_mapping_output)}. Skipping this batch.", file=sys.stderr)

        # Write accumulated results
        if all_mapping_results:
            print(f"Writing {processed_records_count} mapping results to {args.output_file}...")
            with open(args.output_file, 'w', encoding='utf-8') as outfile_json:
                json.dump(all_mapping_results, outfile_json, indent=4)
            print(f"Batch processing complete. {processed_records_count} names processed and results saved to {args.output_file}.")
        elif not (isinstance(batch_mapping_output, dict) and batch_mapping_output.get("is_batch_error")):
             print("No results were successfully processed or collected to write to the output file.")

    except IOError as e:
        print(f"Error writing to output file {args.output_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during main processing loop: {e}", file=sys.stderr)
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}...")
        try:
            shutil.rmtree(temp_dir)
            print(f"Successfully removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error removing temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
    main() 