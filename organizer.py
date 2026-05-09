import os

# MUST be set before ANY other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import shutil
from pathlib import Path
import torch

# Import your classifier
from processing_script import classify_samples

def organize_my_drums(source_dir: str, output_dir: str):
    # Disable PyTorch's internal multithreading which causes Bus Errors on Mac
    torch.set_num_threads(1)
    
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    # 1. Collect files
    file_registry = {}
    extensions = {'.wav', '.aif', '.aiff', '.mp3', '.flac'}
    
    print(f"--- Scanning {source_path} ---")
    for file in source_path.rglob('*'):
        if file.is_file() and file.suffix.lower() in extensions:
            file_registry.setdefault(file.name, []).append(file)

    if not file_registry:
        print("No audio files found!")
        return

    # 2. Run Classification
    # We process in a single main thread to avoid the Bus Error
    print(f"Classifying {len(file_registry)} unique filenames...")
    try:
        results = classify_samples(list(file_registry.keys()))
        
        # 3. Move Files
        print(f"\n--- Moving files to {output_path} ---")
        for res in results:
            target_folder = output_path / res.category
            target_folder.mkdir(parents=True, exist_ok=True)

            for original_path in file_registry[res.raw]:
                dest_path = target_folder / res.raw
                
                # Deduplication logic
                counter = 1
                while dest_path.exists():
                    dest_path = target_folder / f"{Path(res.raw).stem}_{counter}{Path(res.raw).suffix}"
                    counter += 1
                
                shutil.move(str(original_path), str(dest_path))

    except Exception as e:
        print(f"An error occurred during classification: {e}")
    finally:
        print("\nProcess finished.")

if __name__ == "__main__":
    # Final safety: Use 'spawn' instead of 'fork' for macOS stability
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    INPUT_FOLDER = "data/raw"
    OUTPUT_FOLDER = "data/sorted"
    
    organize_my_drums(INPUT_FOLDER, OUTPUT_FOLDER)