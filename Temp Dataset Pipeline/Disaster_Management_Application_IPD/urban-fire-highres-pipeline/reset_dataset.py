import os
import shutil
import glob

def safe_remove_file(filepath):
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return 1
        except Exception:
            pass
    return 0

def reset_dataset():
    stats = {
        'csv_files': 0,
        'sat_images': 0,
        'rejected': 0,
        'cached_scenes': 0,
        'temp_files': 0
    }
    
    # 1. Previous CSV files (input files)
    csv_dir = "data/csv"
    if os.path.exists(csv_dir):
        for f in os.listdir(csv_dir):
            if f.endswith('.csv'):
                stats['csv_files'] += safe_remove_file(os.path.join(csv_dir, f))
    
    # 2. Dataset files (images, metadata, rejected)
    unreviewed_img_dir = "dataset/unreviewed/images"
    if os.path.exists(unreviewed_img_dir):
        for f in os.listdir(unreviewed_img_dir):
            if f.endswith('.jpg') or f.endswith('.png'):
                stats['sat_images'] += safe_remove_file(os.path.join(unreviewed_img_dir, f))
                
    rejected_dir = "dataset/rejected_nonurban"
    if os.path.exists(rejected_dir):
        for root, _, files in os.walk(rejected_dir):
            for f in files:
                if f.endswith('.csv'):
                    stats['rejected'] += safe_remove_file(os.path.join(root, f))
                elif f.endswith('.jpg') or f.endswith('.png'):
                    stats['rejected'] += safe_remove_file(os.path.join(root, f))
                    
    metadata_dir = "dataset/metadata"
    if os.path.exists(metadata_dir):
        for f in os.listdir(metadata_dir):
            if f.endswith('.csv') or f.endswith('.json'):
                stats['temp_files'] += safe_remove_file(os.path.join(metadata_dir, f))
                
    # 3. Cache files (satellite scenes, cache dir)
    cache_dir = "dataset/cache"
    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                stats['cached_scenes'] += safe_remove_file(os.path.join(root, f))
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(os.path.join(cache_dir, "search"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "urban"), exist_ok=True)
            
    # Also clean __pycache__ just in case there are temp files (don't count them towards summary)
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                
    # Logs
    if os.path.exists('pipeline.log'):
        stats['temp_files'] += safe_remove_file('pipeline.log')
        
    print("==================================================")
    print("DATASET RESET")
    print("==================================================")
    print(f"Previous CSV files removed:        {stats['csv_files']:03d}")
    print(f"Previous satellite images removed: {stats['sat_images']:03d}")
    print(f"Rejected images removed:           {stats['rejected']:03d}")
    print(f"Cached scenes removed:             {stats['cached_scenes']:03d}")
    print(f"Temporary files removed:           {stats['temp_files']:03d}")
    print()
    print("Dataset state: CLEAN")
    print("==================================================")

if __name__ == "__main__":
    reset_dataset()
