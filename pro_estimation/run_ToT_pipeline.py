import subprocess
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

scripts_to_run = [
    "pro_estimation/ToT/multi-llms-pro-estimation-ToT-gpt.py",
    "pro_estimation/ToT/multi-llms-pro-estimation-ToT-llama.py",
]

def clean_dot_underscore_files(folder):
    count = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.startswith("._"):
                os.remove(os.path.join(root, f))
                count += 1
    if count:
        print(f"  Removed {count} '._' files in {folder}")

def main():
    print("Starting ToT pipeline (sequential)...\n")

    for script in scripts_to_run:
        script_path = os.path.join(project_root, script)

        if not os.path.exists(script_path):
            print(f"[ERROR] File not found: {script_path}")
            sys.exit(1)

        print("=" * 80)
        print(f"[RUNNING] {script}")
        print("=" * 80)

        try:
            subprocess.run([sys.executable, script_path], check=True)
            print(f"\n[SUCCESS] Finished running {script}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Script {script} failed (exit code {e.returncode})")
            print("Stopping pipeline.")
            sys.exit(e.returncode)

        # Remove ._ files after each script
        output4 = os.path.join(project_root, "output")
        print(f"[CLEAN] Removing '._' files in output...")
        clean_dot_underscore_files(output4)
        print()

    # Final cleanup of the entire project
    print("=" * 80)
    print("[CLEAN] Final removal of '._' files across the project...")
    clean_dot_underscore_files(project_root)

    print("=" * 80)
    print("Entire ToT pipeline completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
