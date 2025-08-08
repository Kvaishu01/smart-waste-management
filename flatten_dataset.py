# backend/flatten_dataset.py

import os
import shutil

BASE_DIR = os.path.join("backend", "data", "waste_dataset", "Garbage classification")

def flatten_category(category):
    category_path = os.path.join(BASE_DIR, category)
    if not os.path.isdir(category_path):
        return

    for root, _, files in os.walk(category_path):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(category_path, file)
            if src != dst:
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    print(f"Error moving {src}: {e}")

    # Remove nested folders
    for item in os.listdir(category_path):
        item_path = os.path.join(category_path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

if __name__ == "__main__":
    for category in os.listdir(BASE_DIR):
        flatten_category(category)
    print("✅ All nested folders flattened.")
