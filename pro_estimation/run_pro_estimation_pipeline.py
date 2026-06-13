import subprocess
import os
import sys

# Get the root directory path of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# List of scripts to run sequentially
scripts_to_run = [
    #"pro_estimation/multi-llms/multi-llms-pro-estimation-few-shot-gpt.py",
    #"pro_estimation/multi-llms/multi-llms-pro-estimation-few-shot-llama.py",
    #"pro_estimation/multi-llms/multi-llms-pro-estimation-zero-shot-gpt.py",
    #"pro_estimation/multi-llms/multi-llms-pro-estimation-zeroshot-llama.py"
    "pro_estimation/one-llm/one-llm-pro-estimation-few-shot-llama.py", 
    "pro_estimation/one-llm/one-llm-pro-estimation-few-shot-gpt.py", 
    "pro_estimation/one-llm/one-llm-pro-estimation-zero-shot-gpt.py",
    "pro_estimation/one-llm/one-llm-pro-estimation-zeroshot-llama.py"

]

def main():
    print("Starting pro_estimation pipeline sequence...\n")
    
    for script in scripts_to_run:
        script_path = os.path.join(project_root, script)
        
        if not os.path.exists(script_path):
            print(f"[ERROR] File not found: {script_path}")
            sys.exit(1)
            
        print("=" * 80)
        print(f"[RUNNING] {script}")
        print("=" * 80)
        
        try:
            # Run script using the current Python interpreter (e.g., inside .venv)
            subprocess.run([sys.executable, script_path], check=True)
            print(f"\n[SUCCESS] Finished running {script}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] An error occurred while running {script}")
            print(f"Exit code: {e.returncode}")
            print("Stopping the entire pipeline.")
            sys.exit(e.returncode)
            
    print("=" * 80)
    print("Entire pipeline completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()