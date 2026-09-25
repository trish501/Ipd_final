import os
import platform
from pathlib import Path
import random
import time

def print_header(msg):
    print("\n" + "="*60)
    print(f" {msg}")
    print("="*60)

def idiot_proof_setup():
    print_header("🔥 DISASTER MANAGEMENT MLOPS SETUP 🔥")
    time.sleep(1)
    
    # 1. Auto-detect OS
    os_name = platform.system()
    print(f"[System Check] Auto-detected Operating System: {os_name}")
    
    # 2. Get Google Drive Path Safely
    print("\n[Action Required]")
    print("Please open your File Explorer (Windows) or Finder (Mac).")
    print("Locate the Master folder on your Google Drive that contains the 2022, 2023, 2024, and 2025 datasets.")
    print("👉 DRAG AND DROP that master folder directly into this terminal window and press ENTER.")
    
    raw_path = input("\nDrag folder here: ").strip()
    
    # Clean up the path (Mac/Windows often wrap dragged paths in quotes or add trailing spaces)
    clean_path = raw_path.strip("'").strip('"').strip()
    gdrive_root = Path(clean_path)
    
    if not gdrive_root.exists():
        print(f"\n❌ ERROR: Could not find the folder at '{clean_path}'")
        print("Please make sure Google Drive is running and you dragged the correct folder.")
        input("Press ENTER to exit...")
        return

    print(f"\n✅ Successfully located dataset root: {gdrive_root}")
    print("\n[Processing] Scanning all 4 years for images... (This may take 10-20 seconds)")
    
    # 3. Compile the Dataset
    all_images = list(gdrive_root.rglob("*.jpg"))
    
    if len(all_images) == 0:
        print("\n❌ ERROR: No .jpg images found! Are you sure you selected the right folder?")
        input("Press ENTER to exit...")
        return
        
    train_paths = []
    val_paths = []
    test_paths = []
    
    for img_path in all_images:
        path_str = str(img_path.absolute())
        
        if "val" in path_str.split(os.sep):
            val_paths.append(path_str)
        elif "test" in path_str.split(os.sep):
            test_paths.append(path_str)
        elif "train" in path_str.split(os.sep) or "industrial_false" in path_str.split(os.sep):
            train_paths.append(path_str)

    # Perfect batch mixing
    random.shuffle(train_paths)
    
    out_dir = Path(__file__).parent
    
    with open(out_dir / "unified_train.txt", "w") as f:
        f.write("\n".join(train_paths))
        
    with open(out_dir / "unified_val.txt", "w") as f:
        f.write("\n".join(val_paths))
        
    with open(out_dir / "unified_test.txt", "w") as f:
        f.write("\n".join(test_paths))
        
    print_header("✅ SETUP COMPLETE!")
    print(f"Successfully linked {len(train_paths) + len(val_paths) + len(test_paths)} total images from Google Drive.")
    print(f"-> Train images: {len(train_paths)}")
    print(f"-> Val images:   {len(val_paths)}")
    print(f"-> Test images:  {len(test_paths)}")
    print("\nYou can now safely close this setup script and run your model's train.py script!")
    input("\nPress ENTER to exit...")

if __name__ == "__main__":
    try:
        idiot_proof_setup()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        input("Press ENTER to exit...")
