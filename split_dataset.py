import os
import shutil
import random

# Set your paths
source_dir = os.path.join("backend", "data", "waste_dataset")
train_dir = os.path.join("backend", "data", "train")
val_dir = os.path.join("backend", "data", "val")

split_ratio = 0.8  # 80% train, 20% validation

# Create train/val directories if they don't exist
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if os.path.isdir(category_path):
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

        images = os.listdir(category_path)
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        val_images = images[split_point:]

        # Copy training images
        for img in train_images:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(train_dir, category, img)
            if os.path.isfile(src_path):
                shutil.copyfile(src_path, dst_path)

        # Copy validation images
        for img in val_images:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(val_dir, category, img)
            if os.path.isfile(src_path):
                shutil.copyfile(src_path, dst_path)

print("✅ Dataset split completed successfully.")
