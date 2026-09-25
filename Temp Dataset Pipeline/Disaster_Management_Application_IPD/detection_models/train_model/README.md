# Distributed Training Module

This isolated folder contains the scripts needed to execute the **Unified Distributed MLOps Plan** for your 4-person team.

## How to use:

1. Mount the shared `Master_Fire_Dataset` Google Drive to your PC.
2. Run the compiler script to generate the unified `.txt` files (this takes seconds and copies zero images to your hard drive):
   ```bash
   python compile_dataset.py --gdrive_path "G:/Master_Fire_Dataset" --output_dir .
   ```
3. You will now see `unified_train.txt`, `unified_val.txt`, and `unified_test.txt` in this folder.
4. Go back to the main `detection_models` directory and run your assigned model's training script, pointing it to `master_data.yaml`:
   ```bash
   python ../yolo11/train.py --dataset train_model/master_data.yaml --model yolo11m.pt
   ```

*(Note: If you decide not to use this Distributed Training approach, you can safely delete this entire `train_model` folder without affecting any of your other pipeline code.)*
