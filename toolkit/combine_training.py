import numpy as np
from pathlib import Path


def process_and_append_datasets(path_A_base, path_B_base, path_C_base, fragment_type):
    path_A = Path(path_A_base) / fragment_type
    path_B = Path(path_B_base)
    path_C = Path(path_C_base)

    path_C.mkdir(parents=True, exist_ok=True)

    file_mappings = {
        'transfer_atn.txt': f'{fragment_type}atn.txt',
        'transfer_e.txt': f'{fragment_type}_e.txt',
        'transfer_f.txt': f'{fragment_type}_f.txt',
        'transfer_xd.txt': f'{fragment_type}_train.txt',
        'transfer_xyz.txt': f'{fragment_type}_train_xyz.txt'
    }

    data_collections = {key: [] for key in file_mappings.keys()}

    # --- 1. Load pre-filtered base files from Path B ---
    if path_B.exists():
        print(f"Loading pre-filtered base files from {path_B}...")
        for file_A, file_B in file_mappings.items():
            base_file_path = path_B / file_B
            if base_file_path.exists():
                data = np.loadtxt(base_file_path, ndmin=2)
                data_collections[file_A].append(data)
                print(f"  Loaded base {file_B} (Shape: {data.shape})")
            else:
                print(f"  Base {file_B} not found.")
    else:
        print(f"Path {path_B} does not exist. Starting fresh with Path A...")

    # --- 2. Scanning and filtering directories in Path A ---
    print(f"\nScanning directories in {path_A}...")
    sub_dirs = sorted(list(set(f.parent for f in path_A.rglob('transfer_*.txt'))))

    for sub_dir in sub_dirs:
        print(f"Processing folder: {sub_dir.name}")

        # Load local order_used.txt to filter the files in this specific folder
        order_path = sub_dir / 'order_used.txt'
        order_indices = None

        if order_path.exists():
            # Flatten to 1D to safely use as numpy array indices
            order_indices = np.loadtxt(order_path, dtype=int, ndmin=1).flatten()
            print(f"  Found order_used.txt: Filtering {len(order_indices)} geometries.")
        else:
            print(f"  No order_used.txt found. Appending all data in folder.")

        # Load and filter standard mapped files
        for file_A in file_mappings.keys():
            target_file = sub_dir / file_A
            if target_file.exists():
                data = np.loadtxt(target_file, ndmin=2)

                # Apply the filter if order_used.txt existed in this folder
                if order_indices is not None:
                    try:
                        data = data[order_indices]
                    except IndexError:
                        print(f"  WARNING: IndexError filtering {file_A}. Check if indices exceed file length!")

                data_collections[file_A].append(data)

    # --- 3. Saving appended files to Path C ---
    print(f"\nSaving appended files to {path_C}...")

    for file_A, file_B in file_mappings.items():
        if data_collections[file_A]:
            final_array = np.vstack(data_collections[file_A])
            save_path = path_C / file_B

            # Determine the saving format: integer for 'atn', 8 decimals for everything else
            if 'atn' in file_B:
                save_fmt = '%i'
            else:
                save_fmt = '%.8f'

            np.savetxt(save_path, final_array, fmt=save_fmt)
            print(f"  Saved {file_B} (Final Shape: {final_array.shape})")


# --- Example Usage ---
path_A = "./"
path_B = "/N/project/sico/xiao/green-onion_0.16/green-onion/ML/training_points/10%_test"
path_C = "./training"
fragment = "H2O1"

process_and_append_datasets(path_A, path_B, path_C, fragment)
