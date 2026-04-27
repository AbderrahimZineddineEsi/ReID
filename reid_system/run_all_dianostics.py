import os
import sys
import subprocess
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path("reid_system/outputs/step2_global_crops_fixed")
SCRIPT_TO_RUN = "reid_system/experiments/diagnose_folder_deep.py"
OUTPUT_LOG_FILE = "all_diagnostics_results.txt"
# ---------------------

def main():
    if not BASE_DIR.exists():
        print(f"Error: Base directory '{BASE_DIR}' does not exist.")
        return

    # Find all subdirectories that start with 'person_' and sort them alphabetically
    person_folders = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("person_")])

    if not person_folders:
        print(f"No person folders found in {BASE_DIR}")
        return

    print(f"Found {len(person_folders)} folders to process. Saving results to {OUTPUT_LOG_FILE}\n")

    # Open the log file in write mode (creates or overwrites the file)
    with open(OUTPUT_LOG_FILE, 'w', encoding='utf-8') as log_file:
        
        for person_dir in person_folders:
            # sys.executable ensures it uses the Python from your active (.venv)
            cmd =[sys.executable, SCRIPT_TO_RUN, "--input", str(person_dir)]
            
            # Create a nice header for the output
            header = f"\n{'='*60}\n"
            header += f"Processing: {person_dir.name}\n"
            header += f"Command: {' '.join(cmd)}\n"
            header += f"{'='*60}\n"
            
            print(header, end="")
            log_file.write(header)

            # Execute the command
            # stdout=subprocess.PIPE and stderr=subprocess.STDOUT merges errors into standard output
            # text=True decodes the byte stream into strings
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Read the output line by line as it is generated (real-time)
            for line in process.stdout:
                # Print to console
                print(line, end="")
                # Write to text file
                log_file.write(line)
                # Flush the file buffer so data is saved immediately
                log_file.flush()

            # Wait for the process to finish
            process.wait()

            footer = f"\n--- Finished {person_dir.name} (Return Code: {process.returncode}) ---\n"
            print(footer)
            log_file.write(footer)

    print(f"\nAll operations completed! Full log saved to: {OUTPUT_LOG_FILE}")

if __name__ == "__main__":
    main()