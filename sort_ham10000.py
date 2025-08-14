import os
import pandas as pd
import shutil

# Base paths
base_path = r'D:\images'
metadata_csv = os.path.join(base_path, 'HAM10000_metadata.csv')
part1_dir = os.path.join(base_path, 'Part 1')
part2_dir = os.path.join(base_path, 'Part 2')
output_dir = os.path.join(base_path, 'sorted_skin_images')

# Read metadata
df = pd.read_csv(metadata_csv)

# List of unique labels (dx column in metadata)
diagnoses = df['dx'].unique()

# Create folders for each diagnosis
for dx in diagnoses:
    os.makedirs(os.path.join(output_dir, dx), exist_ok=True)

possible_exts = ['.jpg', '.JPG', '.jpeg', '.JPEG']

# Copy images to respective folders from both part 1 and part 2
for _, row in df.iterrows():
    image_id = row['image_id']
    label = row['dx']
    src = None

    # Try all extensions in both folders
    for ext in possible_exts:
        candidate1 = os.path.join(part1_dir, image_id + ext)
        candidate2 = os.path.join(part2_dir, image_id + ext)

        if os.path.exists(candidate1):
            src = candidate1
            break
        elif os.path.exists(candidate2):
            src = candidate2
            break

    if src is None:
        print(f"Warning: Image {image_id} not found in either folder with extensions {possible_exts}!")
        continue

    dst = os.path.join(output_dir, label, f'{image_id}.jpg')  # Save all as .jpg
    shutil.copy(src, dst)

print("Images have been sorted by diagnosis.")

# Optional: Print image counts per class
for dx in diagnoses:
    count = len(os.listdir(os.path.join(output_dir, dx)))
    print(f"{dx}: {count} images")
