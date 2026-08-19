import os
import shutil
import subprocess

def clear_directory(dir_path):
    if not os.path.exists(dir_path):
        return
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Starting urban pipeline cleanup...")

    reset_script = os.path.join(base_dir, "reset_dataset.py")
    if os.path.exists(reset_script):
        print("Running reset_dataset.py...")
        subprocess.run(["python", reset_script], cwd=base_dir)

    print("Cleaning up downloaded satellite events...")
    events_dir = os.path.join(base_dir, "dataset", "unreviewed", "events")
    clear_directory(events_dir)

    print("==================================================")
    print("Dataset Pipeline Cleanup Complete!")
    print("==================================================")
