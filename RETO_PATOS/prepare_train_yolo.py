import os
import random
import shutil

base_path = "RETO_PATOS/patos_dataset"
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

# Create train/val folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# Get labeled images only
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt") and os.path.isfile(os.path.join(labels_dir, f))]
random.shuffle(label_files)

# 80/20 split
split_index = int(0.8 * len(label_files))
train_labels = label_files[:split_index]
val_labels = label_files[split_index:]

def move_data(split, label_list):
    for label_file in label_list:
        image_file = label_file.replace(".txt", ".jpg")
        shutil.move(os.path.join(images_dir, image_file), os.path.join(images_dir, split, image_file))
        shutil.move(os.path.join(labels_dir, label_file), os.path.join(labels_dir, split, label_file))

move_data("train", train_labels)
move_data("val", val_labels)

print(f"✅ {len(train_labels)} training images")
print(f"✅ {len(val_labels)} validation images")
